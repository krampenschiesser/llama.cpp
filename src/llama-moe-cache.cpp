#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cinttypes>
#include <condition_variable>
#include <deque>
#include <exception>
#include <functional>
#include <mutex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

// the experts of the layer can be kept in the cache if their plain data can be read from the host
static bool llama_moe_cache_is_cacheable(const llama_model & model, const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        (ggml_backend_buffer_is_host(t->buffer) || model.has_plain_source(t));
}

static uint8_t * llama_moe_cache_align_ptr(void * ptr, size_t alignment) {
    GGML_ASSERT(alignment > 0 && (alignment & (alignment - 1)) == 0);
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    return reinterpret_cast<uint8_t *>((base + alignment - 1) & ~(uintptr_t) (alignment - 1));
}

static uint64_t llama_moe_cache_min_rebalance(int32_t n_expert) {
    const char * env = getenv("LLAMA_MOE_CACHE_REBALANCE");
    if (env != nullptr && *env != '\0') {
        const long long value = atoll(env);
        return value > 0 ? (uint64_t) value : 1;
    }
    // about 32 selections per expert in the epoch, below that the tail of the counts is sampling noise
    return std::max<uint64_t>(8192, 32ull*(uint64_t) n_expert);
}

// an expert that is not cached must be used this much more than the incumbent to take its row
static double llama_moe_cache_hysteresis() {
    const char * env = getenv("LLAMA_MOE_CACHE_HYSTERESIS");
    if (env == nullptr || *env == '\0') {
        return 1.25;
    }
    const double value = atof(env);
    return value >= 1.0 ? value : 1.0;
}

// max experts promoted by one rebalance call, bounds the time the devices stay idle
static uint64_t llama_moe_cache_max_promote() {
    const char * env = getenv("LLAMA_MOE_CACHE_MAX_PROMOTE");
    if (env == nullptr || *env == '\0') {
        return 512;
    }
    const long long value = atoll(env);
    return value > 0 ? (uint64_t) value : 1;
}

// max bytes promoted by one rebalance call, bounds the stall after prompt processing
static size_t llama_moe_cache_max_promote_bytes() {
    const char * env = getenv("LLAMA_MOE_CACHE_MAX_PROMOTE_MIB");
    if (env == nullptr || *env == '\0') {
        return (size_t) 1024*1024*1024;
    }
    const long long value = atoll(env);
    return value > 0 ? (size_t) value*1024*1024 : (size_t) 1;
}

// target duration of a forced rebalance, the byte budget is derived from the measured rate
static int64_t llama_moe_cache_max_promote_us() {
    const char * env = getenv("LLAMA_MOE_CACHE_MAX_PROMOTE_MS");
    if (env == nullptr || *env == '\0') {
        return 250*1000;
    }
    const long long value = atoll(env);
    return value > 0 ? value*1000 : 1;
}

// fixed pool for the upload workers, avoids creating threads on every rebalance
struct llama_moe_cache_pool {
    std::vector<std::thread> workers;
    std::deque<std::function<void()>> tasks;
    std::mutex mutex;
    std::condition_variable cond;
    size_t n_active = 0;
    bool stop = false;
    std::exception_ptr error;

    explicit llama_moe_cache_pool(size_t n) {
        for (size_t i = 0; i < n; ++i) {
            workers.emplace_back([this]() {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        cond.wait(lock, [this]() { return stop || !tasks.empty(); });
                        if (stop && tasks.empty()) {
                            return;
                        }
                        task = std::move(tasks.front());
                        tasks.pop_front();
                        ++n_active;
                    }
                    try {
                        task();
                    } catch (...) {
                        std::lock_guard<std::mutex> lock(mutex);
                        if (!error) {
                            error = std::current_exception();
                        }
                    }
                    std::lock_guard<std::mutex> lock(mutex);
                    --n_active;
                    if (tasks.empty() && n_active == 0) {
                        cond.notify_all();
                    }
                }
            });
        }
    }

    ~llama_moe_cache_pool() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stop = true;
        }
        cond.notify_all();
        for (std::thread & worker : workers) {
            worker.join();
        }
    }

    void run(std::vector<std::function<void()>> jobs) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            for (std::function<void()> & job : jobs) {
                tasks.push_back(std::move(job));
            }
        }
        cond.notify_all();
        std::unique_lock<std::mutex> lock(mutex);
        cond.wait(lock, [this]() { return tasks.empty() && n_active == 0; });
        if (error) {
            // rethrow on the calling thread, a throw inside a worker would terminate the process
            std::exception_ptr err = error;
            error = nullptr;
            lock.unlock();
            std::rethrow_exception(err);
        }
    }
};

// aligned host staging memory, pinned when the device has a host buffer type
struct llama_moe_cache_stage {
    ggml_backend_buffer_ptr buf;
    std::vector<uint8_t> mem;
    uint8_t * ptr = nullptr;

    void alloc(ggml_backend_buffer_type_t buft, size_t bytes, size_t alignment) {
        // the aligned pointer can move up by alignment - 1, keep that on top of the requested size
        const size_t size = bytes + alignment;
        if (buft != nullptr) {
            const size_t alloc = GGML_PAD(size, std::max<size_t>(1, ggml_backend_buft_get_alignment(buft)));
            buf.reset(ggml_backend_buft_alloc_buffer(buft, alloc));
        }
        if (buf) {
            ptr = llama_moe_cache_align_ptr(ggml_backend_buffer_get_base(buf.get()), alignment);
        } else {
            mem.resize(size);
            ptr = llama_moe_cache_align_ptr(mem.data(), alignment);
        }
    }
};

}

// decode one expert plane of a packed host bank back to the plain layout, see the CPU_REPACK buffer
using llama_moe_cache_derepack_fn = bool (*)(const ggml_tensor *, int64_t, void *);

struct llama_moe_cache::impl {
    struct device {
        ggml_backend_t backend = nullptr;
        ggml_backend_buffer_type_t buft = nullptr;
        size_t host_bytes = 0; // bytes of the host experts of the layers it caches


        // device storage of the banks
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // selected experts of the layers of this device, [n_layers*usage_rows_max] I32, one slice per layer
        ggml_tensor * usage = nullptr;
        int32_t usage_n_layers = 0;
        std::vector<int32_t> usage_layers; // layer id of each slice of the usage buffer
        size_t  usage_bytes = 0; // bytes of the buffer read back by one graph

        // expert position of the slot expression, [n_used_max] F32
        ggml_tensor * k_base = nullptr;

        // host slab that receives the usage ids of the graphs in flight, one fixed slot per queued graph
        ggml_backend_buffer_ptr read_slab;
        size_t read_stride = 0;             // bytes of one slot, padded
        std::vector<ggml_backend_event_t> read_ev; // one event per slot
        std::vector<int32_t> read_free;     // indices of the free slots
        std::vector<int32_t> read_rows_cur; // rows of the current graph, by usage slice
        std::vector<uint8_t> read_data;     // host copy when the device has no host buffer
        static constexpr int32_t read_n_slots = 8; // graphs in flight before the queue is drained early
    };

    // host storage of the experts that are not in the device cache, owned by the cache
    struct host_bank {
        ggml_backend_buffer_type_t buft = nullptr;
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;
    };

    struct layer {
        int32_t id      = -1; // device, -1 if the layer is not cached
        int32_t n_cached_experts_per_layer = 0; // experts of the layer kept in the device cache
        int32_t n_zero  = 0;  // zero rows at the end of the banks, the misses read them
        int32_t n_expert = 0;
        int32_t n_host  = 0;  // host slots of the layer, n_expert if the model keeps the host copy
        bool    host_banked = false; // the cache owns the host copy of the experts
        bool    full    = false; // the whole layer is in the device bank, the graph reads it as a plain FFN
        size_t host_bytes = 0;

        std::vector<ggml_tensor *> experts; // model expert tensors, in the order of the banks
        std::vector<ggml_tensor *> banks;   // device storage of the cached experts and of the zero slots
        std::vector<ggml_tensor *> host_banks; // host storage of the experts not cached on a device

        ggml_tensor * slot_map = nullptr; // F32 [1, n_expert] in device memory, slot + 1 if cached
        ggml_tensor * cpu_map  = nullptr; // I32 [1, n_expert] in host memory, host slot if not cached

        // slice of the usage buffer of the device, -1 if the layer has no usage buffer
        int32_t usage_idx = -1;

        std::vector<int32_t> slot_of;      // [n_expert] device slot of the cached experts, -1 if not cached
        std::vector<int32_t> host_slot_of; // [n_expert] host slot of the experts not cached on a device

        layer_info info = {};

        // ranking keeps a long history and forgets a small part of it per rebalance, the trigger
        // counts the selections since the last rebalance
        uint64_t usage_epoch  = 0; // selections since the last rebalance
        uint64_t rebalance_min = 0; // selections of the current epoch needed to rebalance the layer

        uint64_t n_rebalances = 0;
        uint64_t n_promoted   = 0;
        size_t promoted_bytes = 0;

        // usage counters split by whether the selected expert was cached
        uint64_t n_hits   = 0;
        uint64_t n_misses = 0;
    };

    // one read + copy of a run of consecutive experts to a device bank or to a host bank
    struct upload_job {
        int32_t il;
        int32_t ip;
        int32_t src_expert;
        int32_t dst_slot;
        int32_t n;
        int32_t dev; // -1 for the host banks
    };

    // one expert promoted to the device bank slot of an evicted expert, both rows are exchanged
    // without reading the model file: the host bank row is decoded, the device bank row is read back
    struct swap_job {
        int32_t il;
        int32_t ip;
        int32_t expert;    // promoted expert, used as the plain source when the host row cannot be decoded
        int32_t src_slot;  // host bank slot of the promoted expert
        int32_t dev_slot;  // device bank slot of the evicted expert, then of the promoted expert
        int32_t dst_slot;  // host bank slot of the evicted expert
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
        bool full;            // the bank holds every expert of the layer, the ids are expert ids
    };

    const llama_model & model;
    size_t size;                  // device budget
    size_t read_alignment = 1;    // alignment of the plain reads
    int32_t n_ubatch = 0;         // max tokens of a graph, sizes the usage buffers
    int32_t n_used_max = 0;       // expert selections of a token, model wide
    int32_t usage_rows_max = 0;   // n_used_max*n_ubatch, rows of one layer slice of the usage buffer

    std::vector<device> devices;
    std::vector<layer> layers;

    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, ggml_tensor *> host_bindings; // model experts -> host bank

    // host banks, grouped by buffer type
    std::map<ggml_backend_buffer_type_t, host_bank> host_banks;

    // ranking counters, one per expert, they keep a long history
    std::vector<std::vector<uint64_t>> usage;

    // one graph of usage ids in flight: which device, which readback slot, and the rows of its slices
    struct usage_record {
        int32_t dev;
        int32_t slot;
        std::vector<int32_t> rows;
    };
    std::mutex usage_mutex;             // end() pushes, the placement and the stats drain
    std::vector<usage_record> usage_queue;

    // selected experts of the graph being built or of the last built graph, per layer
    mutable std::vector<int32_t> build_rows;
    mutable std::vector<int32_t> last_rows;
    mutable ggml_cgraph * last_gf  = nullptr;
    mutable ggml_cgraph * build_gf = nullptr;
    mutable bool          building = false;

    // optional per graph trace of the bookkeeping cost, see LLAMA_MOE_CACHE_TRACE. it stays in the
    // released code to keep the cost of the cache measurable
    bool     trace = false;
    uint64_t trace_graphs = 0;
    uint64_t trace_calls  = 0;
    uint64_t trace_bytes  = 0;
    uint64_t trace_rows   = 0;
    int64_t  trace_end_us = 0;
    int64_t  trace_count_us = 0;

    // maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    uint64_t n_selected_total  = 0;
    uint64_t n_uploaded_total  = 0;
    size_t uploaded_bytes_total = 0;
    size_t host_filled_bytes_total = 0;
    uint64_t n_rebalances_total = 0;

    double   hysteresis  = 1.25; // an expert must beat the incumbent by this factor to take its row
    uint64_t max_promote = 512;  // experts promoted by one rebalance call
    size_t max_promote_bytes = 1024*1024*1024; // bytes promoted by one rebalance call
    int64_t  max_promote_us  = 250*1000;      // target duration of a forced rebalance
    double   promote_bytes_per_s = 2.0*1024*1024*1024; // measured promotion rate

    std::unique_ptr<llama_moe_cache_pool> pool; // upload workers, created on first use


    std::vector<upload_job> pending_jobs; // filled by the placement code, run by run_uploads()
    std::vector<swap_job> pending_swaps;  // filled by the rebalance, run by run_uploads()

    llama_moe_cache_derepack_fn derepack = nullptr; // host bank inverse, null when the type has no inverse

    // per layer details of the last rebalance, printed by rebalance() after all layers moved
    struct layer_rebalance_stats {
        int32_t il = -1;
        int32_t n_unloaded = 0;
        int32_t n_loaded = 0;
        int32_t n_kept = 0;
        uint64_t n_selections = 0;
        std::vector<std::pair<int32_t, uint64_t>> top; // cached expert id and its selection count
    };
    std::vector<layer_rebalance_stats> last_rebalance_stats;

    // selections that the layers of the last rebalance had counted, also for layers that moved nothing
    uint64_t last_rebalance_selections = 0;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, int32_t n_ubatch) :
            model(model), size(size), n_ubatch(std::max<int32_t>(1, n_ubatch)), layers(model.layers.size()) {
        read_alignment     = std::max<size_t>(1, model.plain_source_alignment());
        hysteresis         = llama_moe_cache_hysteresis();
        max_promote        = llama_moe_cache_max_promote();
        max_promote_bytes  = llama_moe_cache_max_promote_bytes();
        max_promote_us     = llama_moe_cache_max_promote_us();

        n_used_max     = std::max<int32_t>(1, (int32_t) model.hparams.n_expert_used_max());
        usage_rows_max = n_used_max*n_ubatch;
        trace          = getenv("LLAMA_MOE_CACHE_TRACE") != nullptr;

        // the CPU backend can only be a cache device for tests, the real cache needs a GPU
        const bool allow_cpu = getenv("LLAMA_MOE_CACHE_CPU_DEVICE") != nullptr;

        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            const bool is_gpu = dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU;
            if (!is_gpu && !(allow_cpu && dev_type == GGML_BACKEND_DEVICE_TYPE_CPU)) {
                continue;
            }
            auto & d = devices.emplace_back();
            d.backend = backends[i];
            d.buft    = bufts[i];
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || model.hparams.n_expert_used_max() == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }
        if (model.arch == LLM_ARCH_LLAMA4) {
            // the cache slots cannot express the weights that llama4 applies before the FFN
            throw std::runtime_error("MoE cache does not support the llama4 architecture");
        }

        // collect the layers with expert tensors the cache can use
        // a layer is either owned by the cache (the model keeps no host copy) or keeps the host copy in the model
        for (size_t il = 0; il < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty()) {
                continue;
            }
            const bool host_banked = model.is_moe_expert_deferred(experts[0]);
            if (host_banked) {
                bool all_deferred = true;
                bool all_plain = true;
                for (const ggml_tensor * t : experts) {
                    all_deferred = all_deferred && model.is_moe_expert_deferred(t);
                    all_plain = all_plain && model.has_plain_source(t);
                }
                if (!all_deferred || !all_plain) {
                    LLAMA_LOG_WARN("%s: layer %zu has deferred experts that cannot be read from the model file, it is not cached\n", "llama_moe_cache", il);
                    continue;
                }
            } else if (!std::all_of(experts.begin(), experts.end(), [&](const ggml_tensor * t) {
                    return llama_moe_cache_is_cacheable(model, t);
                })) {
                continue;
            }
            // the layout of the expert tensors differs, only the number of experts must match
            const int32_t n_expert = (int32_t) experts[0]->ne[2];
            bool same = n_expert > 0;
            for (const ggml_tensor * t : experts) {
                same = same && (int32_t) t->ne[2] == n_expert;
            }
            if (!same) {
                continue;
            }

            layer & l = layers[il];
            l.experts   = std::move(experts);
            l.n_expert  = n_expert;
            l.host_banked = host_banked;
            // the graph selects the model wide number of experts for every layer, size the zero rows for it
            l.n_zero    = n_used_max;
            for (const ggml_tensor * t : l.experts) {
                l.host_bytes += ggml_nbytes(t);
            }
            // only the layers assigned to a cache device can be cached on a device
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) {
                return ggml_backend_get_device(d.backend) == model.dev_layer(il);
            });
            l.id = it_dev == devices.end() ? -1 : (int32_t) (it_dev - devices.begin());
            // the graph disables the cache for Hadamard-folded weights, keep the full host copy for them
            for (const ggml_tensor * t : l.experts) {
                if (model.hdmd.rot.count(t) > 0) {
                    l.id = -1;
                    break;
                }
            }
            // the cache needs a mul_mat_id over the banks of this layer on the device
            if (l.id >= 0) {
                ggml_backend_dev_t dev = ggml_backend_get_device(devices[l.id].backend);
                for (const ggml_tensor * t : l.experts) {
                    ggml_init_params params = {
                        /*.mem_size   =*/ 8*ggml_tensor_overhead(),
                        /*.mem_buffer =*/ nullptr,
                        /*.no_alloc   =*/ true,
                    };
                    ggml_context_ptr ctx(ggml_init(params));
                    ggml_tensor * w   = ggml_new_tensor_3d(ctx.get(), t->type, t->ne[0], t->ne[1], l.n_expert);
                    ggml_tensor * b   = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, t->ne[0], 1, 1);
                    ggml_tensor * ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, n_used_max, 1);
                    ggml_tensor * op  = ggml_mul_mat_id(ctx.get(), w, b, ids);
                    const bool supported = ggml_backend_dev_supports_op(dev, op);
                    if (!supported) {
                        LLAMA_LOG_WARN("%s: the device cannot compute the experts of layer %zu, it is not cached\n", "llama_moe_cache", il);
                        l.id = -1;
                        break;
                    }
                }
            }
            if (l.id >= 0) {
                devices[l.id].host_bytes += l.host_bytes;
            }
        }

        const bool any_cached = std::any_of(layers.begin(), layers.end(), [](const layer & l) { return l.id >= 0; });
        const bool any_host   = std::any_of(layers.begin(), layers.end(), [](const layer & l) { return l.host_banked; });
        if (!any_cached && !any_host) {
            LLAMA_LOG_WARN("%s: no MoE layer keeps all of its experts in host memory on a cache device, MoE cache is disabled\n", "llama_moe_cache");
            devices.clear();
            return;
        }
        if (devices.empty()) {
            // without a device the cache has nowhere to put the experts, the model experts stay on the CPU
            LLAMA_LOG_WARN("%s: no device to cache the MoE experts, device cache is disabled\n", "llama_moe_cache");
        }

        // one extra row at the end, CUDA MMQ can read past the last expert
        // the size of one expert comes from the buffer type, it may differ from the plain row size
        // when the tensor is repacked, see the IQ3_S/IQ4_XS traits
        auto expert_bytes = [](ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
            const size_t alloc = ggml_backend_buft_get_alloc_size(buft, const_cast<ggml_tensor *>(t));
            return std::max<size_t>(t->nb[2], t->ne[2] > 0 ? alloc/t->ne[2] : t->nb[2]);
        };
        auto alloc_size_n = [expert_bytes](const layer & l, const device & d, int32_t n_cached_experts_per_layer, int32_t n_zero) {
            const size_t alignment = ggml_backend_buft_get_alignment(d.buft);
            size_t res = 0;
            for (const ggml_tensor * t : l.experts) {
                res += GGML_PAD(expert_bytes(d.buft, t)*(n_cached_experts_per_layer + n_zero + 1), alignment);
            }
            return res;
        };
        auto alloc_size = [alloc_size_n](const layer & l, const device & d, int32_t n_cached_experts_per_layer) {
            return alloc_size_n(l, d, n_cached_experts_per_layer, l.n_zero);
        };

        // the budget of each device is clamped to the free memory the cache may use and then raised
        // up to what the layers of the device can hold, so a device with few layers does not strand its share
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });

        const char * env_reserve = getenv("LLAMA_MOE_CACHE_RESERVE_MIB");
        const size_t reserve = env_reserve != nullptr && *env_reserve != '\0'
            ? (size_t) atoll(env_reserve)*1024*1024
            : (size_t) 1536*1024*1024;

        std::vector<double> weight  (devices.size(), 0.0); // requested share, free memory by default
        std::vector<double> capacity(devices.size(), 0.0); // bytes the device can use for its layers
        double weight_sum = 0.0;
        for (size_t id = 0; id < devices.size(); ++id) {
            device & d = devices[id];
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            size_t free_mem  = 0;
            size_t total_mem = 0;
            ggml_backend_dev_memory(dev, &free_mem, &total_mem);

            for (const layer & l : layers) {
                if (l.id == (int32_t) id) {
                    capacity[id] += (double) alloc_size(l, d, std::max<int32_t>(0, l.n_expert - 2));
                }
            }
            if (free_mem > 0) {
                capacity[id] = std::min(capacity[id], free_mem > reserve ? (double) (free_mem - reserve) : 0.0);
            }

            if (d.host_bytes == 0) {
                continue;
            }
            if (split_by_free) {
                // trust the requested size when a device does not report its free memory
                weight[id] = free_mem > 0 ? (double) free_mem : (double) d.host_bytes;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                weight[id] = (double) tensor_split[it - model.devices.begin()];
            }
            weight_sum += weight[id];
        }
        if (weight_sum == 0.0) {
            // the devices do not report their free memory
            for (size_t id = 0; id < devices.size(); ++id) {
                const double dev_bytes = (double) devices[id].host_bytes;
                weight[id] = dev_bytes > 0.0 ? dev_bytes : 0.0;
                weight_sum += weight[id];
            }
        }

        std::vector<double> budget(devices.size(), 0.0);
        {
            std::vector<char> capped(devices.size(), 0);
            double remaining = (double) size;
            for (int iter = 0; iter < (int) devices.size() + 2 && remaining > 0.0; ++iter) {
                double wsum = 0.0;
                for (size_t id = 0; id < devices.size(); ++id) {
                    if (!capped[id] && weight[id] > 0.0) {
                        wsum += weight[id];
                    }
                }
                if (wsum <= 0.0) {
                    break;
                }
                double used = 0.0;
                bool any_capped = false;
                for (size_t id = 0; id < devices.size(); ++id) {
                    if (capped[id] || weight[id] <= 0.0) {
                        continue;
                    }
                    const double share = remaining*weight[id]/wsum;
                    const double room  = capacity[id] - budget[id];
                    if (share >= room) {
                        budget[id] = capacity[id];
                        used += room;
                        capped[id] = 1;
                        any_capped = true;
                    } else {
                        budget[id] += share;
                        used += share;
                    }
                }
                remaining -= used;
                if (!any_capped || remaining <= 0.0) {
                    break;
                }
            }
            double total_budget = 0.0;
            for (const double b : budget) {
                total_budget += b;
            }
            if (total_budget + 1.0 < (double) size) {
                LLAMA_LOG_INFO("%s: MoE cache budget clamped from %.2f MiB to %.2f MiB by the free device memory\n",
                        "llama_moe_cache", size/1024.0/1024.0, total_budget/1024.0/1024.0);
            }
        }

        // within a device the budget is split by the size of the experts, each layer caches its share
        // the layers get dedicated bank rows, as no expert moves during inference
        for (size_t il = 0; il < layers.size(); ++il) {
            layer & l = layers[il];
            if (l.id < 0) {
                continue;
            }
            device & d = devices[l.id];
            const size_t  layer_budget = (size_t) (budget[l.id]*l.host_bytes/d.host_bytes);
            // a layer whose whole expert set fits becomes a plain device FFN: no zero rows, no host
            // bank and no CPU part, so the graph does not build the two halves for it
            if (alloc_size_n(l, d, l.n_expert, 0) <= layer_budget) {
                l.n_cached_experts_per_layer = l.n_expert;
                l.full   = true;
                l.n_zero = 0;
            } else {
                // keep at least 2 experts on the host: the CPU part needs a 3D weight, so the host bank needs 2 planes
                const int32_t n_cached_max = l.n_expert - 2;
                while (l.n_cached_experts_per_layer < n_cached_max && alloc_size(l, d, l.n_cached_experts_per_layer + 1) <= layer_budget) {
                    l.n_cached_experts_per_layer++;
                }
            }
            if (l.n_cached_experts_per_layer < 1) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for layer %zu, it is not cached\n", "llama_moe_cache", il);
                l.id = -1;
                d.host_bytes -= l.host_bytes;
            }
        }
        if (std::none_of(layers.begin(), layers.end(), [](const layer & l) { return l.id >= 0; })) {
            LLAMA_LOG_WARN("%s: MoE cache budget is too small to cache one expert, device cache is disabled\n", "llama_moe_cache");
            devices.clear();
        }

        // the host banks keep n_expert - n_cached experts, the model tensors keep all of them
        for (layer & l : layers) {
            if (l.experts.empty()) {
                continue;
            }
            l.n_host = l.host_banked ? l.n_expert - l.n_cached_experts_per_layer : l.n_expert;
        }

        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        std::map<ggml_backend_buffer_type_t, int> n_tensors_host_bank;
        for (size_t il = 0; il < layers.size(); ++il) {
            layer & l = layers[il];
            if (l.experts.empty()) {
                continue;
            }
            if (l.id >= 0) {
                n_tensors[l.id] += 2*l.experts.size(); // bank and view
                if (!l.full) {
                    n_tensors[l.id] += 1;             // slot map
                    n_tensors_host  += 1;             // CPU map
                    l.usage_idx = devices[l.id].usage_n_layers++;
                    devices[l.id].usage_layers.push_back((int32_t) il);
                    if (l.usage_idx == 0) {
                        n_tensors[l.id] += 2;         // usage buffer and expert position of the device
                    }
                }
            }
            if (l.host_banked && l.n_host > 0) {
                // the bank of a tensor uses the buffer type of that tensor, e.g. the repack buffer for the
                // quant types that have a repack trait
                for (const ggml_tensor * t : l.experts) {
                    n_tensors_host_bank[ggml_backend_buffer_get_type(t->buffer)] += 1;
                }
            }
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        for (device & d : devices) {
            if (d.ctx && d.usage_n_layers > 0) {
                d.usage = ggml_new_tensor_1d(d.ctx.get(), GGML_TYPE_I32, (int64_t) d.usage_n_layers*usage_rows_max);
                ggml_format_name(d.usage, "moe_cache.usage-%d", (int) (&d - devices.data()));
                d.k_base = ggml_new_tensor_1d(d.ctx.get(), GGML_TYPE_F32, n_used_max);
                ggml_format_name(d.k_base, "moe_cache.k_base-%d", (int) (&d - devices.data()));
                const size_t alignment = ggml_backend_buft_get_alignment(d.buft);
                d.usage_bytes = ggml_nbytes(d.usage);
                d.buf_size += GGML_PAD(d.usage_bytes, alignment) + GGML_PAD(ggml_nbytes(d.k_base), alignment);
            }
        }
        if (n_tensors_host > 0) {
            ctx_host = init_ctx(n_tensors_host);
        }
        for (auto & [buft, n] : n_tensors_host_bank) {
            host_bank & b = host_banks[buft];
            b.buft = buft;
            b.ctx  = init_ctx(n);
        }

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t il = 0; il < layers.size(); ++il) {
            layer & l = layers[il];
            if (l.experts.empty()) {
                continue;
            }
            if (l.id >= 0) {
                const device & d = devices[l.id];
                ggml_context * ctx = d.ctx.get();
                for (const ggml_tensor * t : l.experts) {
                    ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], l.n_cached_experts_per_layer + l.n_zero + 1);
                    GGML_ASSERT(bank->nb[2] >= t->nb[2]);
                    ggml_format_name(bank, "moe_cache.%d.%s", (int) il, t->name);
                    l.banks.push_back(bank);

                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], l.n_cached_experts_per_layer + l.n_zero, bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", t->name);
                    bindings[t] = { (int32_t) il, (int32_t) (l.banks.size() - 1), cached, l.full };
                }

                if (!l.full) {
                    l.slot_map = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1, l.n_expert);
                    ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", (int) il);

                    l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, l.n_expert);
                    ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", (int) il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);

                    l.info.backend  = d.backend;
                    l.info.slot_map = l.slot_map;
                    l.info.cpu_map  = l.cpu_map;
                    l.info.k_base   = d.k_base;
                    l.info.n_cached_experts_per_layer  = l.n_cached_experts_per_layer;
                    l.info.n_zero   = l.n_zero;
                    l.info.n_host   = l.n_host;
                    l.info.n_expert = l.n_expert;

                    l.rebalance_min = llama_moe_cache_min_rebalance(l.n_expert);
                }

                devices[l.id].buf_size += alloc_size(l, d, l.n_cached_experts_per_layer);
            }

            if (l.host_banked && l.n_host > 0) {
                for (const ggml_tensor * t : l.experts) {
                    host_bank & b = host_banks.at(ggml_backend_buffer_get_type(t->buffer));
                    // mul_mat_id needs a 3D weight, so the host bank holds at least 2 planes
                    const int32_t n_planes = std::max<int32_t>(l.n_host, 2);
                    ggml_tensor * bank = ggml_new_tensor_3d(b.ctx.get(), t->type, t->ne[0], t->ne[1], n_planes);
                    GGML_ASSERT(bank->nb[2] >= t->nb[2]);
                    ggml_format_name(bank, "moe_cache.host.%d.%s", (int) il, t->name);
                    l.host_banks.push_back(bank);
                    host_bindings[t] = bank;
                }
            }
        }

        usage.resize(layers.size());
        for (size_t il = 0; il < layers.size(); ++il) {
            if (layers[il].id >= 0 && !layers[il].full) {
                usage[il].assign(layers[il].n_expert, 0);
            }
        }


        // with no_alloc, dummy buffers are attached so the memory use can still be measured, see llama_context::memory_breakdown
        auto alloc_bufs = [no_alloc](ggml_context * ctx, ggml_backend_buffer_type_t buft, const char * name, size_t & buf_size) {
            ggml_backend_buffer_ptr buf = llama_memory_bufs::alloc_buf(ctx, buft, no_alloc, name);
            buf_size = no_alloc
                ? ggml_backend_alloc_ctx_tensors_from_buft_size(ctx, buft)
                : ggml_backend_buffer_get_size(buf.get());
            return buf;
        };
        for (device & d : devices) {
            if (!d.ctx) {
                continue;
            }
            d.buf = alloc_bufs(d.ctx.get(), d.buft, "MoE cache", d.buf_size);
            if (!no_alloc) {
                ggml_backend_buffer_clear(d.buf.get(), 0);
            }
        }
        if (ctx_host) {
            buf_host = alloc_bufs(ctx_host.get(), buft_host, "MoE cache CPU maps", buf_host_size);
        }
        if (!no_alloc) {
            // expert position of the slot expression, 0..n_used_max-1
            std::vector<float> k(n_used_max);
            std::iota(k.begin(), k.end(), 0.0f);
            for (device & d : devices) {
                if (d.k_base != nullptr) {
                    ggml_backend_tensor_set(d.k_base, k.data(), 0, k.size()*sizeof(float));
                }
            }
        }
        for (auto & [buft, b] : host_banks) {
            b.buf = alloc_bufs(b.ctx.get(), buft, "MoE cache host banks", b.buf_size);
        }

        if (!host_banks.empty()) {
            ggml_backend_dev_t dev_cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
            ggml_backend_reg_t reg = dev_cpu != nullptr ? ggml_backend_dev_backend_reg(dev_cpu) : nullptr;
            if (reg != nullptr) {
                derepack = (llama_moe_cache_derepack_fn) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_repack_derepack");
            }
        }

        if (!no_alloc) {

            // the initial placement is the first experts of each layer
            const int64_t t_start_us = ggml_time_us();
            for (size_t il = 0; il < layers.size(); ++il) {
                if (!layers[il].experts.empty()) {
                    set_initial(layers[il], (int32_t) il);
                }
            }
            run_uploads(std::move(pending_jobs), {});
            LLAMA_LOG_INFO("%s: initial placement took %.2f s\n", "llama_moe_cache", (ggml_time_us() - t_start_us)/1e6);
        }

        // as weights, the ops that read the banks run on the device
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        if (buf_host) {
            ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        for (auto & [buft, b] : host_banks) {
            if (b.buf) {
                ggml_backend_buffer_set_usage(b.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }

        size_t host_bank_bytes = 0;
        size_t host_expert_bytes = 0;
        for (const auto & [buft, b] : host_banks) {
            host_bank_bytes += b.buf_size;
        }
        for (const layer & l : layers) {
            host_expert_bytes += l.host_bytes;
        }
        if (host_bank_bytes > 0) {
            LLAMA_LOG_INFO("%s: %10s MoE cache host banks = %8.2f MiB for %.2f MiB of all experts (%.1f%%)\n", "llama_moe_cache",
                ggml_backend_buft_name(host_banks.begin()->first), host_bank_bytes/1024.0/1024.0,
                host_expert_bytes/1024.0/1024.0, 100.0*host_bank_bytes/host_expert_bytes);
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", "llama_moe_cache",
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (size_t il = 0; il < layers.size(); ++il) {
                const layer & l = layers[il];
                if (l.id != (int32_t) id) {
                    continue;
                }
                if (l.full) {
                    LLAMA_LOG_INFO("%s: %2d layer, %s: whole layer cached\n", "llama_moe_cache",
                        (int) il, ggml_type_name(l.experts[0]->type));
                    continue;
                }
                LLAMA_LOG_INFO("%s: %2d layer, %s: %5d cached experts, %d zero rows (%.1f%%)\n", "llama_moe_cache",
                    (int) il, ggml_type_name(l.experts[0]->type), l.n_cached_experts_per_layer, l.n_zero,
                    100.0*l.n_cached_experts_per_layer/l.n_expert);
            }
        }
        LLAMA_LOG_INFO("%s: initial placement of %.2f MiB from the plain source, %.2f MiB to the host banks\n", "llama_moe_cache",
            uploaded_bytes_total/1024.0/1024.0, host_filled_bytes_total/1024.0/1024.0);
    }

    ~impl() {
        drain_usage();
        for (device & d : devices) {
            for (ggml_backend_event_t & ev : d.read_ev) {
                if (ev != nullptr) {
                    ggml_backend_event_free(ev);
                    ev = nullptr;
                }
            }
        }
        log_stats();
    }

    // write the maps of the layer from slot_of and host_slot_of
    void write_maps(layer & l) {
        if (l.cpu_map == nullptr) {
            // the layer is not cached, the CPU map is not used
            return;
        }
        std::vector<float> slots(l.n_expert, 0.0f);
        int32_t * cpu = (int32_t *) l.cpu_map->data;
        for (int32_t e = 0; e < l.n_expert; ++e) {
            if (l.slot_of[e] >= 0) {
                slots[e] = (float) (l.slot_of[e] + 1);
                cpu[e]   = -1;
            } else {
                slots[e] = 0.0f;
                cpu[e]   = l.host_banked ? l.host_slot_of[e] : e;
            }
        }
        if (l.slot_map != nullptr) {
            ggml_backend_tensor_set(l.slot_map, slots.data(), 0, slots.size()*sizeof(float));
        }
    }

    // copy one expert of the host bank to the plain layout in dst, false when it must come from the file
    bool read_host_expert(const layer & l, size_t ip, int32_t slot, uint8_t * dst, size_t size) {
        const ggml_tensor * w = l.host_banks[ip];
        if (w->extra == nullptr) {
            if (!ggml_backend_buffer_is_host(w->buffer)) {
                return false;
            }
            memcpy(dst, (const uint8_t *) w->data + (size_t) slot*w->nb[2], size);
            return true;
        }
        if (derepack == nullptr) {
            return false;
        }
        return derepack(w, slot, dst);
    }

    // read n experts of tensor ip starting at src_expert, the data starts at *delta in stage
    bool read_experts_staged(const layer & l, size_t ip, int32_t src_expert, size_t n, uint8_t * stage, size_t stage_len, size_t * delta) {
        const ggml_tensor * w = l.experts[ip];
        const size_t expert_size = w->nb[2];
        const size_t offs = (size_t) src_expert*expert_size;
        const size_t size = n*expert_size;
        *delta = 0;
        if (!l.host_banked && ggml_backend_buffer_is_host(w->buffer)) {
            memcpy(stage, (const uint8_t *) w->data + offs, size);
            return true;
        }
        return model.read_plain_source_staged(w, offs, size, stage, stage_len, delta);
    }

    // run the jobs on the fixed pool: the host banks in parallel, then one worker per device bank so
    // the copies stay ordered, then one worker per device and expert tensor for the swaps so the
    // derepack and the repack of the three expert tensors run in parallel
    void run_uploads(std::vector<upload_job> jobs, std::vector<swap_job> swaps) {
        if (jobs.empty() && swaps.empty()) {
            return;
        }
        if (!pool) {
            pool = std::make_unique<llama_moe_cache_pool>(16);
        }

        std::map<int32_t, std::vector<const upload_job *>> groups;
        for (const upload_job & job : jobs) {
            groups[job.dev].push_back(&job);
        }
        // the host banks are written by several workers, the repack buffer is slow and they write
        // disjoint slots; the device banks use one worker per device, the copies must stay ordered
        std::vector<std::function<void()>> upload_tasks;
        for (auto & [dev, list] : groups) {
            if (dev >= 0) {
                upload_tasks.push_back([this, dev, list]() { upload_group(dev, list); });
                continue;
            }
            const size_t n_workers = std::min<size_t>(16, std::max<size_t>(1, list.size()));
            for (size_t k = 0; k < n_workers; ++k) {
                std::vector<const upload_job *> part;
                for (size_t i = k; i < list.size(); i += n_workers) {
                    part.push_back(list[i]);
                }
                upload_tasks.push_back([this, part]() { upload_group(-1, part); });
            }
        }
        if (!upload_tasks.empty()) {
            pool->run(std::move(upload_tasks));
        }

        if (swaps.empty()) {
            return;
        }

        // the three expert tensors of a swap go to different bank rows, their derepack and repack can
        // run in parallel, the device calls are serialized per device
        std::map<std::pair<int32_t, int32_t>, std::vector<const swap_job *>> swap_groups;
        for (const swap_job & job : swaps) {
            swap_groups[{ layers[job.il].id, job.ip }].push_back(&job);
        }
        std::map<int32_t, std::unique_ptr<std::mutex>> dev_mutex;
        std::vector<std::function<void()>> swap_tasks;
        for (auto & [key, list] : swap_groups) {
            const int32_t dev = key.first;
            GGML_ASSERT(dev >= 0);
            if (dev_mutex.count(dev) == 0) {
                dev_mutex[dev] = std::make_unique<std::mutex>();
            }
            std::mutex * mutex = dev_mutex[dev].get();
            swap_tasks.push_back([this, dev, list, mutex]() { swap_group(dev, list, *mutex); });
        }
        pool->run(std::move(swap_tasks));
    }

    void upload_group(int32_t dev, const std::vector<const upload_job *> & jobs) {
        const size_t alignment = std::max<size_t>(1, read_alignment);

        ggml_backend_t backend = dev >= 0 ? devices[dev].backend : nullptr;
        ggml_backend_buffer_type_t buft_host = backend != nullptr
            ? ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend))
            : nullptr;

        size_t max_bytes = 0;
        for (const upload_job * job : jobs) {
            max_bytes = std::max(max_bytes, (size_t) job->n*layers[job->il].experts[job->ip]->nb[2]);
        }
        if (max_bytes == 0) {
            return;
        }

        // a direct read of [offs, offs + size) can touch up to 2*alignment extra bytes
        const size_t stage_len = max_bytes + 2*alignment;

        ggml_backend_event_t events[2] = { nullptr, nullptr };
        llama_moe_cache_stage stage[2];
        for (int k = 0; k < 2; ++k) {
            stage[k].alloc(buft_host, stage_len, alignment);
            if (backend != nullptr) {
                events[k] = ggml_backend_event_new(ggml_backend_get_device(backend));
            }
        }

        for (size_t k = 0; k < jobs.size(); ++k) {
            const upload_job * job = jobs[k];
            const int buf = (int) (k & 1);
            layer & l = layers[job->il];
            const size_t size = (size_t) job->n*l.experts[job->ip]->nb[2];

            if (events[buf] != nullptr) {
                ggml_backend_event_synchronize(events[buf]);
            }

            size_t delta = 0;
            if (!read_experts_staged(l, job->ip, job->src_expert, job->n, stage[buf].ptr, stage_len, &delta)) {
                GGML_ABORT("the MoE cache cannot read the plain source of expert tensor %s", l.experts[job->ip]->name);
            }

            if (backend != nullptr) {
                ggml_tensor * dst = l.banks[job->ip];
                const size_t dst_off = (size_t) job->dst_slot*dst->nb[2];
                ggml_backend_tensor_set_async(backend, dst, stage[buf].ptr + delta, dst_off, size);
                if (events[buf] != nullptr) {
                    ggml_backend_event_record(events[buf], backend);
                }
            } else {
                ggml_tensor * dst = l.host_banks[job->ip];
                ggml_backend_tensor_set(dst, stage[buf].ptr + delta, (size_t) job->dst_slot*dst->nb[2], size);
            }
        }

        for (int k = 0; k < 2; ++k) {
            if (events[k] != nullptr) {
                ggml_backend_event_synchronize(events[k]);
                ggml_backend_event_free(events[k]);
            }
        }
    }

    // exchange the rows of the promoted and of the evicted experts of one expert tensor. the device
    // calls of a device are serialized by dev_mutex, the derepack and the host repack run outside it
    void swap_group(int32_t dev, const std::vector<const swap_job *> & jobs, std::mutex & dev_mutex) {
        const size_t alignment = std::max<size_t>(1, read_alignment);
        ggml_backend_t backend = devices[dev].backend;
        ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));

        size_t max_bytes = 0;
        for (const swap_job * job : jobs) {
            max_bytes = std::max(max_bytes, (size_t) layers[job->il].banks[job->ip]->nb[2]);
        }

        const size_t stage_len = max_bytes + 2*alignment;

        ggml_backend_event_t events_h2d[2] = { nullptr, nullptr };
        ggml_backend_event_t events_d2h[2] = { nullptr, nullptr };
        llama_moe_cache_stage stage[2];
        llama_moe_cache_stage evict[2];
        for (int k = 0; k < 2; ++k) {
            stage[k].alloc(buft_host, stage_len, alignment);
            evict[k].alloc(buft_host, max_bytes, alignment);
            events_h2d[k] = ggml_backend_event_new(ggml_backend_get_device(backend));
            if (buft_host != nullptr) {
                events_d2h[k] = ggml_backend_event_new(ggml_backend_get_device(backend));
            }
        }

        const size_t n_swap = jobs.size();

        // start the read of the evicted row of swap k, the promoted row is not needed yet
        auto evict_start = [&](size_t k) {
            const swap_job * job = jobs[k];
            layer & l = layers[job->il];
            ggml_tensor * dst = l.banks[job->ip];
            const int buf = (int) (k & 1);
            const size_t off  = (size_t) job->dev_slot*dst->nb[2];
            const size_t size = dst->nb[2];
            if (events_d2h[buf] != nullptr) {
                ggml_backend_event_synchronize(events_d2h[buf]);
                std::lock_guard<std::mutex> lock(dev_mutex);
                ggml_backend_tensor_get_async(backend, dst, evict[buf].ptr, off, size);
                ggml_backend_event_record(events_d2h[buf], backend);
            } else {
                std::lock_guard<std::mutex> lock(dev_mutex);
                ggml_backend_tensor_get(dst, evict[buf].ptr, off, size);
            }
        };

        evict_start(0);

        for (size_t k = 0; k < n_swap; ++k) {
            const swap_job * job = jobs[k];
            const int buf = (int) (k & 1);
            layer & l = layers[job->il];
            ggml_tensor * dst  = l.banks[job->ip];
            ggml_tensor * host = l.host_banks[job->ip];
            const size_t size = dst->nb[2];

            if (k + 1 < n_swap) {
                evict_start(k + 1);
            }

            // the evicted expert leaves the device bank before the promoted expert takes its row
            if (events_d2h[buf] != nullptr) {
                ggml_backend_event_synchronize(events_d2h[buf]);
            }
            if (events_h2d[buf] != nullptr) {
                ggml_backend_event_synchronize(events_h2d[buf]);
            }

            size_t delta = 0;
            if (!read_host_expert(l, job->ip, job->src_slot, stage[buf].ptr, size) &&
                    !read_experts_staged(l, job->ip, job->expert, 1, stage[buf].ptr, stage_len, &delta)) {
                GGML_ABORT("the MoE cache cannot read the host copy of expert tensor %s", l.experts[job->ip]->name);
            }

            {
                std::lock_guard<std::mutex> lock(dev_mutex);
                ggml_backend_tensor_set_async(backend, dst, stage[buf].ptr + delta, (size_t) job->dev_slot*dst->nb[2], size);
                if (events_h2d[buf] != nullptr) {
                    ggml_backend_event_record(events_h2d[buf], backend);
                }
            }
            ggml_backend_tensor_set(host, evict[buf].ptr, (size_t) job->dst_slot*host->nb[2], size);
        }

        for (int k = 0; k < 2; ++k) {
            if (events_h2d[k] != nullptr) {
                ggml_backend_event_synchronize(events_h2d[k]);
                std::lock_guard<std::mutex> lock(dev_mutex);
                ggml_backend_event_free(events_h2d[k]);
            }
            if (events_d2h[k] != nullptr) {
                std::lock_guard<std::mutex> lock(dev_mutex);
                ggml_backend_event_free(events_d2h[k]);
            }
        }
    }

    // queue the copy of n consecutive experts of tensor ip to the device bank rows at dst_slot
    void add_upload(layer & l, int32_t il, int32_t ip, int32_t src_expert, int32_t dst_slot, int32_t n) {
        const size_t expert_size = l.experts[ip]->nb[2];
        // the counters are per expert, the caller queues one job per expert tensor
        if (ip == 0) {
            l.n_promoted     += n;
            n_uploaded_total += n;
        }
        l.promoted_bytes     += n*expert_size;
        uploaded_bytes_total += n*expert_size;
        pending_jobs.push_back({ il, ip, src_expert, dst_slot, n, l.id });
    }

    // queue the fill of n consecutive experts to the host bank rows at dst_slot
    void add_host_fill(layer & l, int32_t il, int32_t src_expert, int32_t dst_slot, int32_t n) {
        if (!l.host_banked) {
            return;
        }
        const size_t expert_size = l.experts[0]->nb[2];
        const int32_t per_chunk = (int32_t) std::max<size_t>(1, 32u*1024*1024/expert_size);
        for (int32_t i = 0; i < n; i += per_chunk) {
            const int32_t k = std::min(per_chunk, n - i);
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                pending_jobs.push_back({ il, (int32_t) ip, src_expert + i, dst_slot + i, k, -1 });
                host_filled_bytes_total += (size_t) k*l.experts[ip]->nb[2];
            }
        }
    }

    // queue the fill of every expert of the layer that is not in the device cache
    void add_host_fill_all(layer & l, int32_t il) {
        if (!l.host_banked) {
            return;
        }
        int32_t e = 0;
        while (e < l.n_expert) {
            if (l.host_slot_of[e] < 0) {
                e++;
                continue;
            }
            int32_t k = 1;
            while (e + k < l.n_expert && l.host_slot_of[e + k] == l.host_slot_of[e] + k) {
                k++;
            }
            add_host_fill(l, il, e, l.host_slot_of[e], k);
            e += k;
        }
    }

    // merge the uploads of the experts when consecutive experts go to consecutive bank rows
    void add_uploads_merged(layer & l, int32_t il, const std::vector<int32_t> & experts) {
        // bound the staging buffer of a job, the read and the copy of a large run stay in the cache
        size_t expert_size = 1;
        for (const ggml_tensor * t : l.experts) {
            expert_size = std::max(expert_size, t->nb[2]);
        }
        const size_t per_chunk = std::max<size_t>(1, 32u*1024*1024/expert_size);

        for (size_t i = 0; i < experts.size(); ) {
            size_t k = 1;
            while (i + k < experts.size() &&
                    experts[i + k] == experts[i] + (int32_t) k &&
                    l.slot_of[experts[i + k]] == l.slot_of[experts[i]] + (int32_t) k) {
                k++;
            }
            for (size_t j = 0; j < k; j += per_chunk) {
                const int32_t n = (int32_t) std::min(per_chunk, k - j);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    add_upload(l, il, (int32_t) ip, experts[i + j], l.slot_of[experts[i + j]], n);
                }
            }
            i += k;
        }
    }

    void set_initial(layer & l, int32_t il) {
        l.slot_of.assign(l.n_expert, -1);
        l.host_slot_of.assign(l.n_expert, -1);

        // the cache starts with the first experts of the layer
        std::vector<int32_t> cached_ids;
        for (int32_t e = 0; e < l.n_cached_experts_per_layer; ++e) {
            cached_ids.push_back(e);
        }
        for (size_t i = 0; i < cached_ids.size(); ++i) {
            l.slot_of[cached_ids[i]] = (int32_t) i;
        }
        if (l.host_banked) {
            int32_t slot = 0;
            for (int32_t e = 0; e < l.n_expert; ++e) {
                if (l.slot_of[e] < 0) {
                    l.host_slot_of[e] = slot++;
                }
            }
        }
        write_maps(l);

        if (l.id < 0) {
            // all experts are read on the CPU, the host banks are the only copy
            add_host_fill_all(l, il);
            return;
        }
        add_uploads_merged(l, il, cached_ids);
        add_host_fill_all(l, il);
    }

    // the most used experts of the layer replace the least used ones, bounded by the promotion budget
    void rebalance_layer(size_t il, uint64_t & budget) {
        layer & l = layers[il];
        if (l.id < 0) {
            return;
        }
        const int32_t n        = l.n_expert;
        const int32_t n_cached = l.n_cached_experts_per_layer;

        auto count = [&](int32_t e) {
            return usage[il][e];
        };

        // rank the experts by selections, retained experts win the ties
        std::vector<int32_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
            const uint32_t ca = count(a);
            const uint32_t cb = count(b);
            if (ca != cb) {
                return ca > cb;
            }
            const bool ha = l.slot_of[a] >= 0;
            const bool hb = l.slot_of[b] >= 0;
            if (ha != hb) {
                return ha;
            }
            return a < b;
        });

        // the desired set is the n_cached most used experts
        std::vector<char> in_sel(n, 0);
        for (int32_t i = 0; i < n_cached; ++i) {
            in_sel[order[i]] = 1;
        }
        // retained experts outside the desired set would be evicted, weakest first
        std::vector<int32_t> evicted;
        std::vector<int32_t> promoted;
        for (int32_t i = n - 1; i >= 0; --i) {
            const int32_t e = order[i];
            if (l.slot_of[e] >= 0 && !in_sel[e]) {
                evicted.push_back(e);
            }
        }
        for (int32_t i = 0; i < n_cached; ++i) {
            if (l.slot_of[order[i]] < 0) {
                promoted.push_back(order[i]);
            }
        }
        GGML_ASSERT(promoted.size() == evicted.size());

        // hysteresis: a promoted expert must be clearly better than the expert it evicts, otherwise the
        // incumbent keeps its row and the promotion waits for a later rebalance. promoted and evicted are
        // both sorted by count, so the first pair that fails ends the loop
        size_t n_swap = 0;
        while (n_swap < promoted.size() &&
                (double) count(promoted[n_swap]) > hysteresis*(double) count(evicted[n_swap])) {
            n_swap++;
        }
        n_swap = (size_t) std::min<uint64_t>(n_swap, budget);

        // keep the rows of the retained experts, the evicted experts that are not replaced keep theirs
        std::vector<int32_t> final_sel;
        final_sel.reserve(n_cached);
        for (int32_t i = 0; i < n_cached; ++i) {
            if (l.slot_of[order[i]] >= 0) {
                final_sel.push_back(order[i]);
            }
        }
        for (size_t k = 0; k < n_swap; ++k) {
            final_sel.push_back(promoted[k]);
        }
        for (size_t k = n_swap; k < evicted.size(); ++k) {
            final_sel.push_back(evicted[k]);
        }
        GGML_ASSERT((int32_t) final_sel.size() == n_cached);

        // give the freed bank rows to the promoted experts, each promoted expert takes the row of
        // one evicted expert so that the two rows of a swap are known
        std::vector<int32_t> new_slot_of(n, -1);
        std::vector<bool> used(n_cached, false);
        for (int32_t e : final_sel) {
            const int32_t s = l.slot_of[e];
            if (s >= 0) {
                new_slot_of[e] = s;
                used[s]        = true;
            }
        }
        std::vector<int32_t> expert_of_slot(n_cached, -1);
        for (int32_t e = 0; e < n; ++e) {
            if (l.slot_of[e] >= 0) {
                expert_of_slot[l.slot_of[e]] = e;
            }
        }
        std::vector<int32_t> free_slots;
        for (int32_t s = 0; s < n_cached; ++s) {
            if (!used[s]) {
                free_slots.push_back(s);
            }
        }
        std::vector<int32_t> new_experts;
        for (int32_t e : final_sel) {
            if (new_slot_of[e] < 0) {
                new_experts.push_back(e);
            }
        }
        GGML_ASSERT(new_experts.size() == free_slots.size());
        std::sort(new_experts.begin(), new_experts.end());

        std::vector<std::pair<int32_t, int32_t>> swaps; // promoted expert, evicted expert
        for (size_t i = 0; i < new_experts.size(); ++i) {
            const int32_t slot = free_slots[i];
            const int32_t e    = expert_of_slot[slot];
            GGML_ASSERT(e >= 0 && new_slot_of[e] < 0);
            new_slot_of[new_experts[i]] = slot;
            swaps.emplace_back(new_experts[i], e);
        }
        budget -= new_experts.size();

        // the host banks of the cache keep the experts that are not on the device
        // the promoted experts free their host slots, the evicted experts take them
        std::vector<int32_t> new_host_slot_of;
        if (l.host_banked) {
            std::vector<bool> evicted_of(n, false);
            for (const auto & [p, e] : swaps) {
                evicted_of[e] = true;
            }
            new_host_slot_of.assign(n, -1);
            for (int32_t e = 0; e < n; ++e) {
                if (new_slot_of[e] < 0 && !evicted_of[e]) {
                    GGML_ASSERT(l.host_slot_of[e] >= 0);
                    new_host_slot_of[e] = l.host_slot_of[e];
                }
            }
            for (const auto & [p, e] : swaps) {
                GGML_ASSERT(l.host_slot_of[p] >= 0);
                new_host_slot_of[e] = l.host_slot_of[p];
            }
            for (int32_t e = 0; e < n; ++e) {
                if (new_slot_of[e] < 0) {
                    GGML_ASSERT(new_host_slot_of[e] >= 0);
                }
            }
        }

        // keep the details for the idle log, the counters are decayed below
        layer_rebalance_stats st;
        st.il = (int32_t) il;
        st.n_loaded = (int32_t) new_experts.size();
        int32_t n_old_cached = 0;
        int32_t n_kept = 0;
        for (int32_t e = 0; e < n; ++e) {
            if (l.slot_of[e] >= 0) {
                n_old_cached += 1;
                if (new_slot_of[e] >= 0) {
                    n_kept += 1;
                }
            }
        }
        st.n_unloaded = n_old_cached - n_kept;
        st.n_kept     = n_kept;
        st.n_selections = l.usage_epoch;
        for (int32_t i = 0; i < std::min<int32_t>(10, n_cached); ++i) {
            st.top.emplace_back(order[i], usage[il][order[i]]);
        }
        if (!new_experts.empty()) {
            last_rebalance_stats.push_back(std::move(st));
        }

        l.slot_of.swap(new_slot_of);
        if (l.host_banked) {
            l.host_slot_of.swap(new_host_slot_of);
        }
        if (!new_experts.empty()) {
            write_maps(l);
        }

        if (l.host_banked) {
            // exchange the rows of the promoted and of the evicted experts without reading the model file
            for (const auto & [p, e] : swaps) {
                const int32_t slot = l.host_slot_of[e];
                l.n_promoted    += 1;
                n_uploaded_total += 1;
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    const size_t expert_size = l.experts[ip]->nb[2];
                    l.promoted_bytes     += expert_size;
                    uploaded_bytes_total += expert_size;
                    host_filled_bytes_total += expert_size;
                    pending_swaps.push_back({ (int32_t) il, (int32_t) ip, p, slot, l.slot_of[p], slot });
                }
            }
        } else {
            // queue the uploads of the promoted experts, merged when consecutive
            add_uploads_merged(l, (int32_t) il, new_experts);
        }
    }

    void count_ids(size_t il, const int32_t * ids, size_t n) {
        layer & l = layers[il];
        // a full layer has no usage buffer and no counters
        if (l.id < 0 || l.full) {
            return;
        }
        const int32_t n_experts = l.n_expert;
        for (size_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e >= 0 && e < n_experts) {
                usage[il][e] += 1;
                if (l.slot_of[e] >= 0) {
                    l.n_hits += 1;
                } else {
                    l.n_misses += 1;
                }
            }
        }
        l.usage_epoch  += n;
        n_selected_total += n;
    }

    // count the selected experts of the layers of the device from a host copy of the usage buffer
    void count_rows(device & d, const int32_t * base, const int32_t * rows_of) {
        const int64_t t0 = trace ? ggml_time_us() : 0;
        for (size_t idx = 0; idx < d.usage_layers.size(); ++idx) {
            if (rows_of[idx] > 0) {
                count_ids(d.usage_layers[idx], base + idx*usage_rows_max, rows_of[idx]);
            }
        }
        if (trace) {
            trace_count_us += ggml_time_us() - t0;
        }
    }

    // host pointer of a readback slot of the device
    const void * read_slot_ptr(const device & d, int32_t slot) const {
        return (const uint8_t *) ggml_backend_buffer_get_base(d.read_slab.get()) + (size_t) slot*d.read_stride;
    }

    // make the readback slab of the device, allocated once and reused by the slots
    void init_read_slab(device & d, ggml_backend_buffer_type_t buft_host) {
        if (d.read_slab) {
            return;
        }
        const size_t alignment = std::max<size_t>(1, ggml_backend_buft_get_alignment(buft_host));
        d.read_stride = GGML_PAD(d.usage_bytes, alignment);
        d.read_slab.reset(ggml_backend_buft_alloc_buffer(buft_host, d.read_stride*device::read_n_slots));
        if (!d.read_slab) {
            throw std::runtime_error("failed to allocate the MoE cache read buffer");
        }
        d.read_ev.resize(device::read_n_slots, nullptr);
        for (int32_t i = device::read_n_slots - 1; i >= 0; --i) {
            d.read_free.push_back(i);
        }
    }

    // count the graph held by a readback slot of the device, the event may still be in flight
    void count_read(device & d, int32_t slot, const std::vector<int32_t> & rows_of) {
        if (d.read_ev[slot] != nullptr) {
            ggml_backend_event_synchronize(d.read_ev[slot]);
        } else {
            ggml_backend_synchronize(d.backend);
        }
        count_rows(d, (const int32_t *) read_slot_ptr(d, slot), rows_of.data());
    }

    // count the graphs that are queued and release their buffers, called when the placement is decided
    // or when the counters are read. no expert moves here
    void drain_usage() {
        std::lock_guard<std::mutex> lock(usage_mutex);
        const int64_t t0 = trace ? ggml_time_us() : 0;
        for (const usage_record & r : usage_queue) {
            device & d = devices[r.dev];
            count_read(d, r.slot, r.rows);
            d.read_free.push_back(r.slot);
        }
        usage_queue.clear();
        if (trace) {
            trace_count_us += ggml_time_us() - t0;
        }
    }

    // count the reads that are still in flight
    void finish_reads() {
        drain_usage();
    }

    // rebalance the layers that collected enough selections, or all of them when forced, bounded by
    // the promotion budget. the budget starts with an equal share per layer so that high layer ids
    // do not starve, the leftover is used in a second pass
    uint64_t rebalance_layers(bool force, uint64_t max_experts, size_t max_bytes) {
        std::vector<size_t> candidates;
        std::vector<char>   touched(layers.size(), 0);
        for (size_t il = 0; il < layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.id < 0 || l.full) {
                continue;
            }
            // a forced rebalance skips the layers without new selections, re-sorting them with the
            // same counters only decays the history
            const uint64_t min = force ? 1 : l.rebalance_min;
            if (l.usage_epoch < min) {
                continue;
            }
            candidates.push_back(il);
        }
        if (candidates.empty()) {
            return 0;
        }

        uint64_t experts_left = max_experts;
        size_t   bytes_left   = max_bytes;
        uint64_t moved        = 0;

        auto run = [&](uint64_t experts_cap, size_t bytes_cap) {
            for (size_t il : candidates) {
                if (experts_left == 0 || bytes_left == 0) {
                    break;
                }
                layer & l = layers[il];
                const size_t bytes_per_expert = std::max<size_t>(1, l.host_bytes/std::max(1, l.n_expert));
                uint64_t allowed = std::min<uint64_t>(experts_left,
                        std::min<uint64_t>(experts_cap, (uint64_t) (bytes_cap/bytes_per_expert)));
                if (allowed == 0) {
                    continue;
                }
                const uint64_t before = allowed;
                rebalance_layer(il, allowed);
                touched[il] = 1;
                const uint64_t m = before - allowed;
                moved        += m;
                experts_left -= m;
                bytes_left   -= std::min<size_t>(bytes_left, m*bytes_per_expert);
            }
        };

        run(std::max<uint64_t>(1, max_experts/candidates.size()), std::max<size_t>(1, max_bytes/candidates.size()));
        run(experts_left, bytes_left);

        // slow decay of the ranking history, once per rebalanced layer: the two passes above must see
        // the same counters and the epoch restarts only now
        last_rebalance_selections = 0;
        for (size_t il : candidates) {
            if (!touched[il]) {
                continue;
            }
            layer & l = layers[il];
            last_rebalance_selections += l.usage_epoch;
            for (int32_t e = 0; e < l.n_expert; ++e) {
                usage[il][e] -= usage[il][e]/8;
            }
            l.usage_epoch      = 0;
            l.n_rebalances    += 1;
            n_rebalances_total += 1;
        }

        return moved;
    }

    const layer_info * get_layer(int32_t il) const {
        // a full layer is a plain device FFN, the graph does not need the cache slots
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].id < 0 || layers[il].full) {
            return nullptr;
        }
        return &layers[il].info;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        // the slots of a partial bank index n_cached_experts_per_layer + n_zero rows, only a full
        // bank can be read with plain expert ids, see get_full_experts
        return it != bindings.end() && !it->second.full ? it->second.cached : nullptr;
    }

    ggml_tensor * get_full_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() && it->second.full ? it->second.cached : nullptr;
    }

    ggml_tensor * get_host_experts(const ggml_tensor * w) const {
        const auto it = host_bindings.find(w);
        return it != host_bindings.end() ? it->second : nullptr;
    }

    llama_moe_cache_stats stats() const {
        llama_moe_cache_stats res = {};
        res.enabled = !devices.empty() || !host_banks.empty();
        for (const layer & l : layers) {
            if (l.id < 0) {
                continue;
            }
            res.n_layers_cached  += 1;
            res.n_experts_cached += l.n_cached_experts_per_layer;
            res.n_hits           += l.n_hits;
            res.n_misses         += l.n_misses;
            res.n_promoted       += l.n_promoted;
            res.bytes_promoted   += l.promoted_bytes;
        }
        res.n_selected   = n_selected_total;
        res.n_rebalances = n_rebalances_total;
        res.bytes_uploaded = uploaded_bytes_total;
        for (const device & d : devices) {
            if (d.buf) {
                const size_t size = ggml_backend_buffer_get_size(d.buf.get());
                res.bytes_device += size > 0 ? size : d.buf_size;
            }
        }
        for (const auto & [buft, b] : host_banks) {
            if (b.buf) {
                const size_t size = ggml_backend_buffer_get_size(b.buf.get());
                res.bytes_host += size > 0 ? size : b.buf_size;
            }
        }
        if (buf_host) {
            const size_t size = ggml_backend_buffer_get_size(buf_host.get());
            res.bytes_host += size > 0 ? size : buf_host_size;
        }
        return res;
    }

    // apply the usage counters while the devices are idle, e.g. after all server slots are released
    // when forced the per layer threshold is ignored, e.g. to apply the usage of a prompt before the decode
    void rebalance(bool force) {
        const uint64_t rebalances_before = n_rebalances_total;
        const uint64_t uploaded_before   = n_uploaded_total;
        const size_t   bytes_before      = uploaded_bytes_total;

        for (device & d : devices) {
            ggml_backend_synchronize(d.backend);
        }
        finish_reads();

        // a forced rebalance runs between a prompt and the decode, bound its duration with the measured
        // promotion rate. the idle rebalance has no one waiting, only the byte budget applies
        size_t bytes_budget = max_promote_bytes;
        if (force) {
            const size_t timed = (size_t) (promote_bytes_per_s*(double) max_promote_us/1e6);
            bytes_budget = std::min(bytes_budget, std::max<size_t>(1, timed));
        }

        // only the layers that collected enough selections since the last rebalance move their experts
        last_rebalance_stats.clear();
        const int64_t t_start_us = ggml_time_us();
        const uint64_t moved = rebalance_layers(force, max_promote, bytes_budget);
        run_uploads(std::move(pending_jobs), std::move(pending_swaps));
        const double elapsed_s = (ggml_time_us() - t_start_us)/1e6;

        // track the promotion rate for the time bound of the next forced rebalance
        const size_t bytes_moved = uploaded_bytes_total - bytes_before;
        if (bytes_moved > 0 && elapsed_s > 1e-4 && (uploaded_bytes_total > bytes_before)) {
            const double rate = (double) bytes_moved/elapsed_s;
            promote_bytes_per_s = 0.5*promote_bytes_per_s + 0.5*rate;
        }

        if (n_rebalances_total > rebalances_before) {
            uint64_t n_loaded_experts = 0;
            for (const auto & st : last_rebalance_stats) {
                n_loaded_experts += (uint64_t) st.n_loaded;
            }
            LLAMA_LOG_INFO("llama_moe_cache: rebalance%s: %" PRIu64 " layer(s), %" PRIu64 " experts promoted, "
                    "%" PRIu64 " expert uploads (%.2f MiB, %.2f MiB/s), %" PRIu64 " selections counted, %.2f s\n",
                force ? " (forced)" : " (idle)",
                n_rebalances_total - rebalances_before, n_loaded_experts,
                n_uploaded_total - uploaded_before, (uploaded_bytes_total - bytes_before)/1024.0/1024.0,
                (uploaded_bytes_total - bytes_before)/1024.0/1024.0/std::max(1e-6, elapsed_s),
                last_rebalance_selections, elapsed_s);
            for (const auto & st : last_rebalance_stats) {
                std::string top;
                char buf[32];
                for (const auto & t : st.top) {
                    snprintf(buf, sizeof(buf), top.empty() ? "%d(%" PRIu64 ")" : " %d(%" PRIu64 ")", (int) t.first, t.second);
                    top += buf;
                }
                LLAMA_LOG_DEBUG("llama_moe_cache:   layer %2d: unloaded %2d, loaded %2d, kept %2d, top remaining ids: %s\n",
                    (int) st.il, (int) st.n_unloaded, (int) st.n_loaded, (int) st.n_kept, top.c_str());
            }
        }
        GGML_UNUSED(moved);
    }

    // the device storage that receives the selected experts of layer il for the next graph
    ggml_tensor * get_usage(ggml_context * ctx, ggml_cgraph * gf, int32_t il, int64_t n_used, int64_t n_tokens) const {
        const layer & l = layers[il];
        if (l.id < 0 || l.usage_idx < 0) {
            return nullptr;
        }
        const int64_t rows = n_used*n_tokens;
        if (rows <= 0 || rows > usage_rows_max) {
            return nullptr;
        }
        if (build_rows.empty()) {
            build_rows.assign(layers.size(), 0);
        }
        build_rows[il] = (int32_t) rows;
        build_gf       = gf;

        const size_t offset = (size_t) l.usage_idx*usage_rows_max*sizeof(int32_t);
        ggml_tensor * res = ggml_view_1d(ctx, devices[l.id].usage, rows, offset);
        ggml_format_name(res, "moe_cache.usage-%d", (int) il);
        return res;
    }

    // called after the graph compute is queued: read the selected experts back for the counters
    void end(ggml_cgraph * gf) {
        // a reused graph selects the same rows as when it was built
        const bool built = building && gf == build_gf;
        if (!built && gf != last_gf) {
            return;
        }
        const std::vector<int32_t> & rows_of = built ? build_rows : last_rows;
        if (rows_of.empty()) {
            return;
        }
        const int64_t t0 = trace ? ggml_time_us() : 0;

        size_t rows_read  = 0;
        size_t bytes_read = 0;
        for (device & d : devices) {
            if (d.usage == nullptr) {
                continue;
            }
            // rows of the layers of this device in this graph, by usage slice
            d.read_rows_cur.resize(d.usage_layers.size());
            size_t rows = 0;
            for (size_t idx = 0; idx < d.usage_layers.size(); ++idx) {
                d.read_rows_cur[idx] = rows_of[d.usage_layers[idx]];
                rows = std::max<size_t>(rows, (size_t) d.read_rows_cur[idx]);
            }
            if (rows == 0) {
                continue;
            }
            rows_read += rows;

            ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(d.backend));
            if (buft_host == nullptr) {
                // no host buffer to read into without blocking
                if (d.read_data.size() < d.usage_bytes) {
                    d.read_data.resize(d.usage_bytes);
                }
                ggml_backend_tensor_get(d.usage, d.read_data.data(), 0, d.usage_bytes);
                count_rows(d, (const int32_t *) d.read_data.data(), d.read_rows_cur.data());
                continue;
            }
            // the usage buffer holds the rows of every graph in the same layout, so the whole buffer is
            // read back with one call into a slot of the slab. the record is counted later, at the
            // placement or when the stats are read, so the submit thread does not wait for the readback
            init_read_slab(d, buft_host);
            int32_t slot = -1;
            {
                std::lock_guard<std::mutex> lock(usage_mutex);
                if (!d.read_free.empty()) {
                    slot = d.read_free.back();
                    d.read_free.pop_back();
                }
            }
            if (slot < 0) {
                // all slots are in flight, count the queued graphs now
                drain_usage();
                std::lock_guard<std::mutex> lock(usage_mutex);
                slot = d.read_free.back();
                d.read_free.pop_back();
            }
            if (d.read_ev[slot] == nullptr) {
                d.read_ev[slot] = ggml_backend_event_new(ggml_backend_get_device(d.backend));
            }
            if (d.read_ev[slot] == nullptr) {
                // the backend has no events, count right away
                ggml_backend_tensor_get(d.usage, const_cast<void *>(read_slot_ptr(d, slot)), 0, d.usage_bytes);
                count_rows(d, (const int32_t *) read_slot_ptr(d, slot), d.read_rows_cur.data());
                std::lock_guard<std::mutex> lock(usage_mutex);
                d.read_free.push_back(slot);
                continue;
            }
            ggml_backend_tensor_get_async(d.backend, d.usage, const_cast<void *>(read_slot_ptr(d, slot)), 0, d.usage_bytes);
            ggml_backend_event_record(d.read_ev[slot], d.backend);
            {
                std::lock_guard<std::mutex> lock(usage_mutex);
                usage_queue.push_back({ (int32_t) (&d - devices.data()), slot, d.read_rows_cur });
            }

            bytes_read += d.usage_bytes;
            if (trace) {
                trace_calls++;
                trace_bytes += d.usage_bytes;
            }
        }

        if (trace) {
            trace_graphs++;
            trace_rows   += rows_read;
            trace_end_us += ggml_time_us() - t0;
            LLAMA_LOG_DEBUG("llama_moe_cache: graph %d nodes, %zu rows, %zu B read back, end %.0f us\n",
                ggml_graph_n_nodes(gf), rows_read, bytes_read, (double) (ggml_time_us() - t0));
        }
        if (built) {
            last_rows = build_rows;
            last_gf   = gf;
            building  = false;
            build_gf  = nullptr;
        }
    }

    // called before a graph is built, the selected rows of the new graph replace the cached ones
    void new_graph() {
        building = true;
        last_gf  = nullptr;
        build_gf = nullptr;
        build_rows.clear();
    }

    void log_stats() const {
        if (trace && trace_graphs > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: trace: %" PRIu64 " graphs, %" PRIu64 " readbacks (%.2f MiB, %" PRIu64 " rows), "
                    "end %.2f ms (%.1f us/graph), count %.2f ms\n",
                trace_graphs, trace_calls, trace_bytes/1024.0/1024.0, trace_rows,
                trace_end_us/1000.0, (double) trace_end_us/std::max<uint64_t>(1, trace_graphs),
                trace_count_us/1000.0);
        }
        if (n_selected_total == 0) {
            // a cache that holds whole layers computes them as a plain FFN, so it counts no selections
            int32_t n_full = 0;
            for (const layer & l : layers) {
                n_full += l.full ? 1 : 0;
            }
            if (n_full > 0) {
                LLAMA_LOG_INFO("llama_moe_cache: %d layer(s) are cached in full and compute no misses\n", (int) n_full);
            }
            return;
        }
        uint64_t n_hits   = 0;
        uint64_t n_misses = 0;
        for (const layer & l : layers) {
            n_hits   += l.n_hits;
            n_misses += l.n_misses;
        }
        LLAMA_LOG_INFO("llama_moe_cache: counted %" PRIu64 " selected experts, %" PRIu64 " hits (%.1f%%), %" PRIu64 " misses, %" PRIu64 " rebalances, %" PRIu64 " experts uploaded (%.2f MiB)\n",
            n_selected_total, n_hits, 100.0*(double) n_hits/std::max<uint64_t>(1, n_hits + n_misses), n_misses,
            n_rebalances_total, n_uploaded_total, uploaded_bytes_total/1024.0/1024.0);
        for (size_t il = 0; il < layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.id < 0 || l.n_rebalances == 0) {
                continue;
            }
            LLAMA_LOG_INFO("llama_moe_cache: layer %2d: %d cached experts, %" PRIu64 " hits (%.1f%%), %" PRIu64 " misses, %" PRIu64 " rebalances, %" PRIu64 " experts promoted (%.2f MiB)\n",
                (int) il, l.n_cached_experts_per_layer, l.n_hits, 100.0*(double) l.n_hits/std::max<uint64_t>(1, l.n_hits + l.n_misses),
                l.n_misses, l.n_rebalances, l.n_promoted, l.promoted_bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, int32_t n_ubatch) :
    pimpl(new impl(model, no_alloc, backends, bufts, size, n_ubatch)) {
}

llama_moe_cache::~llama_moe_cache() = default;

const llama_moe_cache::layer_info * llama_moe_cache::get_layer(int32_t il) const {
    return pimpl->get_layer(il);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

ggml_tensor * llama_moe_cache::get_full_experts(const ggml_tensor * w) const {
    return pimpl->get_full_experts(w);
}

ggml_tensor * llama_moe_cache::get_host_experts(const ggml_tensor * w) const {
    return pimpl->get_host_experts(w);
}

ggml_tensor * llama_moe_cache::get_usage(ggml_context * ctx, ggml_cgraph * gf, int32_t il, int64_t n_used, int64_t n_tokens) const {
    return pimpl->get_usage(ctx, gf, il, n_used, n_tokens);
}

void llama_moe_cache::end(ggml_cgraph * gf) {
    pimpl->end(gf);
}

void llama_moe_cache::new_graph() {
    pimpl->new_graph();
}

void llama_moe_cache::finish() {
    pimpl->finish_reads();
}

void llama_moe_cache::rebalance() {
    pimpl->rebalance(true);
}

void llama_moe_cache::rebalance_idle() {
    pimpl->rebalance(false);
}

llama_moe_cache_stats llama_moe_cache::stats() const {
    return pimpl->stats();
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            const size_t size = ggml_backend_buffer_get_size(d.buf.get());
            res[ggml_backend_buffer_get_type(d.buf.get())] += size > 0 ? size : d.buf_size;
        }
    }
    for (const auto & [buft, b] : pimpl->host_banks) {
        if (b.buf) {
            const size_t size = ggml_backend_buffer_get_size(b.buf.get());
            res[buft] += size > 0 ? size : b.buf_size;
        }
    }
    if (pimpl->buf_host) {
        const size_t size = ggml_backend_buffer_get_size(pimpl->buf_host.get());
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += size > 0 ? size : pimpl->buf_host_size;
    }
    return res;
}
