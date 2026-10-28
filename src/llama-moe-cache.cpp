#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-memory.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
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
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);
    return reinterpret_cast<uint8_t *>((base + alignment - 1) & ~(uintptr_t) (alignment - 1));
}

static uint64_t llama_moe_cache_min_rebalance(int32_t n_expert) {
    const char * env = getenv("LLAMA_MOE_CACHE_REBALANCE");
    if (env != nullptr && *env != '\0') {
        const long long value = atoll(env);
        return value > 0 ? (uint64_t) value : 1;
    }
    // about 32 selections per expert, below that the tail of the usage counts is sampling noise
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

// optional placement profile, seeded at startup and written after each rebalance
static std::string llama_moe_cache_profile_path() {
    const char * env = getenv("LLAMA_MOE_CACHE_PROFILE");
    if (env == nullptr || *env == '\0' || strcmp(env, "0") == 0) {
        return {};
    }
    return env;
}

}

struct llama_moe_cache::impl {
    struct device {
        ggml_backend_t backend = nullptr;
        ggml_backend_buffer_type_t buft = nullptr;
        size_t host_bytes = 0; // bytes of the host experts of the layers it caches


        // device storage of the banks
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // the graphs using the cache queued a read of the selected experts since the last begin()
        bool dirty = false;
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
        size_t host_bytes = 0;

        std::vector<ggml_tensor *> experts; // model expert tensors, in the order of the banks
        std::vector<ggml_tensor *> banks;   // device storage of the cached experts and of the zero slots
        std::vector<ggml_tensor *> host_banks; // host storage of the experts not cached on a device

        ggml_tensor * slot_map = nullptr; // I32 [1, n_expert] in host memory, slot + 1 if cached
        ggml_tensor * cpu_map  = nullptr; // I32 [1, n_expert] in host memory, host slot if not cached

        std::vector<int32_t> slot_of;      // [n_expert] device slot of the cached experts, -1 if not cached
        std::vector<int32_t> host_slot_of; // [n_expert] host slot of the experts not cached on a device

        layer_info info;

        uint64_t usage_total  = 0; // selected experts counted since the last rebalance
        uint64_t rebalance_min = 0; // selections needed to rebalance the layer

        // readback of the selected experts
        ggml_backend_buffer_ptr read_buf;
        size_t read_buf_size = 0;
        bool read_pending = false;
        size_t read_nbytes = 0;
        std::vector<int32_t> read_data; // used when the device has no host buffer

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

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    const llama_model & model;
    size_t size;                  // device budget
    size_t read_alignment = 1;    // alignment of the plain reads

    std::vector<device> devices;
    std::vector<layer> layers;

    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer
    std::unordered_map<const ggml_tensor *, ggml_tensor *> host_bindings; // model experts -> host bank

    // host banks, grouped by buffer type
    std::map<ggml_backend_buffer_type_t, host_bank> host_banks;

    // usage counters, one array per layer, incremented without synchronization
    std::vector<std::unique_ptr<std::atomic<uint32_t>[]>> usage;

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
    std::string profile_path;
    struct {
        std::vector<std::vector<int32_t>> cached; // cached expert ids per layer
        bool loaded = false;
    } profile;


    std::vector<upload_job> pending_jobs; // filled by the placement code, run by run_uploads()

    // per layer details of the last rebalance, printed by rebalance() after all layers moved
    struct layer_rebalance_stats {
        int32_t il = -1;
        int32_t n_unloaded = 0;
        int32_t n_loaded = 0;
        int32_t n_kept = 0;
        uint64_t n_selections = 0;
        std::vector<std::pair<int32_t, uint32_t>> top; // cached expert id and its selection count
    };
    std::vector<layer_rebalance_stats> last_rebalance_stats;

    impl(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
            model(model), size(size), layers(model.layers.size()) {
        read_alignment     = std::max<size_t>(1, model.plain_source_alignment());
        hysteresis         = llama_moe_cache_hysteresis();
        max_promote        = llama_moe_cache_max_promote();
        profile_path       = llama_moe_cache_profile_path();

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
                    LLAMA_LOG_WARN("%s: layer %zu has deferred experts that cannot be read from the model file, it is not cached\n", __func__, il);
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
            l.n_zero    = (int32_t) (il < model.hparams.n_layer_all ? model.hparams.n_expert_used(il) : model.hparams.n_expert_used_max());
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
            if (l.id >= 0) {
                devices[l.id].host_bytes += l.host_bytes;
            }
        }

        const bool any_cached = std::any_of(layers.begin(), layers.end(), [](const layer & l) { return l.id >= 0; });
        const bool any_host   = std::any_of(layers.begin(), layers.end(), [](const layer & l) { return l.host_banked; });
        if (!any_cached && !any_host) {
            LLAMA_LOG_WARN("%s: no MoE layer keeps all of its experts in host memory on a cache device, MoE cache is disabled\n", __func__);
            devices.clear();
            return;
        }
        if (devices.empty()) {
            // without a device the cache has nowhere to put the experts, the model experts stay on the CPU
            LLAMA_LOG_WARN("%s: no device to cache the MoE experts, device cache is disabled\n", __func__);
        }

        // one extra row at the end, CUDA MMQ can read past the last expert
        // the size of one expert comes from the buffer type, it may differ from the plain row size
        // when the tensor is repacked, see the IQ3_S/IQ4_XS traits
        auto expert_bytes = [](ggml_backend_buffer_type_t buft, const ggml_tensor * t) {
            const size_t alloc = ggml_backend_buft_get_alloc_size(buft, const_cast<ggml_tensor *>(t));
            return std::max<size_t>(t->nb[2], t->ne[2] > 0 ? alloc/t->ne[2] : t->nb[2]);
        };
        auto alloc_size = [expert_bytes](const layer & l, const device & d, int32_t n_cached_experts_per_layer) {
            const size_t alignment = ggml_backend_buft_get_alignment(d.buft);
            size_t res = 0;
            for (const ggml_tensor * t : l.experts) {
                res += GGML_PAD(expert_bytes(d.buft, t)*(n_cached_experts_per_layer + l.n_zero + 1), alignment);
            }
            return res;
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
            capacity[id] = std::min(capacity[id], free_mem > reserve ? (double) (free_mem - reserve) : 0.0);

            if (d.host_bytes == 0) {
                continue;
            }
            if (split_by_free) {
                weight[id] = (double) free_mem;
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
                weight[id] = devices[id].host_bytes > 0 ? 1.0 : 0.0;
                weight_sum += weight[id];
            }
        }

        std::vector<double> budget(devices.size(), 0.0);
        {
            std::vector<char> capped(devices.size(), 0);
            double remaining = (double) size;
            for (int iter = 0; iter < 16 && remaining > 0.0; ++iter) {
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
                        __func__, size/1024.0/1024.0, total_budget/1024.0/1024.0);
            }
        }

        // within a device the budget is split by the size of the experts, each layer caches its share
        // the layers get dedicated bank rows, as no expert moves during inference
        for (size_t il = 0; il < layers.size(); ++il) {
            layer & l = layers[il];
            if (l.id < 0) {
                continue;
            }
            const device & d = devices[l.id];
            const size_t  layer_budget = (size_t) (budget[l.id]*l.host_bytes/d.host_bytes);
            // keep at least 2 experts on the host: the CPU part needs a 3D weight, so the host bank needs 2 planes
            const int32_t n_cached_max = l.n_expert - 2;
            while (l.n_cached_experts_per_layer < n_cached_max && alloc_size(l, d, l.n_cached_experts_per_layer + 1) <= layer_budget) {
                l.n_cached_experts_per_layer++;
            }
            if (l.n_cached_experts_per_layer < 1) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for layer %zu, it is not cached\n", __func__, il);
                l.id = -1;
            }
        }
        if (std::none_of(layers.begin(), layers.end(), [](const layer & l) { return l.id >= 0; })) {
            LLAMA_LOG_WARN("%s: MoE cache budget is too small to cache one expert, device cache is disabled\n", __func__);
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
        for (layer & l : layers) {
            if (l.experts.empty()) {
                continue;
            }
            if (l.id >= 0) {
                n_tensors[l.id] += 2*l.experts.size(); // bank and view
                n_tensors_host  += 2;                 // slot map and CPU map
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
                    bindings[t] = { (int32_t) il, (int32_t) (l.banks.size() - 1), cached };
                }

                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, l.n_expert);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", (int) il);
                layer_of[l.slot_map] = (int32_t) il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);

                l.cpu_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, l.n_expert);
                ggml_format_name(l.cpu_map, "moe_cache.cpu_map-%d", (int) il);
                buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_map), alignment_host);

                l.info.backend  = d.backend;
                l.info.slot_map = l.slot_map;
                l.info.cpu_map  = l.cpu_map;
                l.info.n_cached_experts_per_layer  = l.n_cached_experts_per_layer;
                l.info.n_zero   = l.n_zero;
                l.info.n_host   = l.n_host;
                l.info.n_expert = l.n_expert;

                l.rebalance_min = llama_moe_cache_min_rebalance(l.n_expert);

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
            if (layers[il].id >= 0) {
                const int32_t n = layers[il].n_expert;
                usage[il].reset(new std::atomic<uint32_t>[n]);
                for (int32_t e = 0; e < n; ++e) {
                    usage[il][e].store(0, std::memory_order_relaxed);
                }
            }
        }

        load_profile();

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
            buf_host = alloc_bufs(ctx_host.get(), buft_host, "MoE cache slot maps", buf_host_size);
        }
        for (auto & [buft, b] : host_banks) {
            b.buf = alloc_bufs(b.ctx.get(), buft, "MoE cache host banks", b.buf_size);
        }

        if (!no_alloc) {

            // the initial placement comes from the profile when it matches, or from the first experts
            const int64_t t_start_us = ggml_time_us();
            for (size_t il = 0; il < layers.size(); ++il) {
                if (!layers[il].experts.empty()) {
                    set_initial(layers[il], (int32_t) il);
                }
            }
            run_uploads(std::move(pending_jobs));
            LLAMA_LOG_INFO("%s: initial placement took %.2f s\n", __func__, (ggml_time_us() - t_start_us)/1e6);
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
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
            LLAMA_LOG_INFO("%s: %10s MoE cache host banks = %8.2f MiB for %.2f MiB of all experts (%.1f%%)\n", __func__,
                ggml_backend_buft_name(host_banks.begin()->first), host_bank_bytes/1024.0/1024.0,
                host_expert_bytes/1024.0/1024.0, 100.0*host_bank_bytes/host_expert_bytes);
        }

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (size_t il = 0; il < layers.size(); ++il) {
                const layer & l = layers[il];
                if (l.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2d layer, %s: %5d cached experts, %d zero rows (%.1f%%)\n", __func__,
                        (int) il, ggml_type_name(l.experts[0]->type), l.n_cached_experts_per_layer, l.n_zero,
                        100.0*l.n_cached_experts_per_layer/l.n_expert);
                }
            }
        }
        LLAMA_LOG_INFO("%s: initial placement of %.2f MiB from the plain source, %.2f MiB to the host banks\n", __func__,
            uploaded_bytes_total/1024.0/1024.0, host_filled_bytes_total/1024.0/1024.0);
    }

    ~impl() {
        save_profile();
        log_stats();
    }

    // write the host maps of the layer from slot_of and host_slot_of
    void write_maps(layer & l) {
        int32_t * slots = (int32_t *) l.slot_map->data;
        int32_t * cpu   = (int32_t *) l.cpu_map->data;
        for (int32_t e = 0; e < l.n_expert; ++e) {
            if (l.slot_of[e] >= 0) {
                slots[e] = l.slot_of[e] + 1;
                cpu[e]   = -1;
            } else {
                slots[e] = 0;
                cpu[e]   = l.host_banked ? l.host_slot_of[e] : e;
            }
        }
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

    // one worker per target (device bank or host banks), each with two staging buffers so the read of
    // the next job starts while the copy of the current one is in flight
    void run_uploads(std::vector<upload_job> jobs) {
        if (jobs.empty()) {
            return;
        }
        std::map<int32_t, std::vector<const upload_job *>> groups;
        for (const upload_job & job : jobs) {
            groups[job.dev].push_back(&job);
        }
        // the host banks are written by several workers, the repack buffer is slow and they write
        // disjoint slots; the device banks use one worker per device, the copies must stay ordered
        std::vector<std::pair<int32_t, std::vector<const upload_job *>>> queues;
        for (auto & [dev, list] : groups) {
            if (dev >= 0) {
                queues.emplace_back(dev, std::move(list));
                continue;
            }
            const size_t n_workers = std::min<size_t>(8, std::max<size_t>(1, list.size()));
            for (size_t k = 0; k < n_workers; ++k) {
                std::vector<const upload_job *> part;
                for (size_t i = k; i < list.size(); i += n_workers) {
                    part.push_back(list[i]);
                }
                queues.emplace_back(-1, std::move(part));
            }
        }
        std::vector<std::thread> workers;
        workers.reserve(queues.size());
        for (auto & [dev, list] : queues) {
            workers.emplace_back([this, dev, list]() { upload_group(dev, list); });
        }
        for (std::thread & worker : workers) {
            worker.join();
        }
    }

    void upload_group(int32_t dev, const std::vector<const upload_job *> & jobs) {
        size_t max_bytes = 0;
        for (const upload_job * job : jobs) {
            max_bytes = std::max(max_bytes, (size_t) job->n*layers[job->il].experts[job->ip]->nb[2]);
        }
        const size_t alignment = std::max<size_t>(1, read_alignment);
        const size_t buf_size  = max_bytes + 2*alignment;

        ggml_backend_t backend = dev >= 0 ? devices[dev].backend : nullptr;
        ggml_backend_event_t events[2] = { nullptr, nullptr };
        std::vector<uint8_t> storage[2];
        uint8_t * stage[2] = { nullptr, nullptr };
        for (int k = 0; k < 2; ++k) {
            storage[k].resize(buf_size);
            stage[k] = llama_moe_cache_align_ptr(storage[k].data(), alignment);
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
            if (!read_experts_staged(l, job->ip, job->src_expert, job->n, stage[buf], buf_size, &delta)) {
                GGML_ABORT("the MoE cache cannot read the plain source of expert tensor %s", l.experts[job->ip]->name);
            }

            if (backend != nullptr) {
                ggml_tensor * dst = l.banks[job->ip];
                const size_t dst_off = (size_t) job->dst_slot*dst->nb[2];
                ggml_backend_tensor_set_async(backend, dst, stage[buf] + delta, dst_off, size);
                if (events[buf] != nullptr) {
                    ggml_backend_event_record(events[buf], backend);
                }
            } else {
                ggml_tensor * dst = l.host_banks[job->ip];
                ggml_backend_tensor_set(dst, stage[buf] + delta, (size_t) job->dst_slot*dst->nb[2], size);
            }
        }
        for (int k = 0; k < 2; ++k) {
            if (events[k] != nullptr) {
                ggml_backend_event_synchronize(events[k]);
                ggml_backend_event_free(events[k]);
            }
        }
    }

    // queue the copy of n consecutive experts of tensor ip to the device bank rows at dst_slot
    void add_upload(layer & l, int32_t il, int32_t ip, int32_t src_expert, int32_t dst_slot, int32_t n) {
        const size_t expert_size = l.experts[ip]->nb[2];
        l.n_promoted         += n;
        l.promoted_bytes     += n*expert_size;
        n_uploaded_total     += n;
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
        for (size_t i = 0; i < experts.size(); ) {
            size_t k = 1;
            while (i + k < experts.size() &&
                    experts[i + k] == experts[i] + (int32_t) k &&
                    l.slot_of[experts[i + k]] == l.slot_of[experts[i]] + (int32_t) k) {
                k++;
            }
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                add_upload(l, il, (int32_t) ip, experts[i], l.slot_of[experts[i]], (int32_t) k);
            }
            i += k;
        }
    }

    bool profile_layer(size_t il, const layer & l) const {
        return profile.loaded && il < profile.cached.size() &&
            (int32_t) profile.cached[il].size() == l.n_cached_experts_per_layer;
    }

    void load_profile() {
        if (profile_path.empty()) {
            return;
        }
        std::ifstream in(profile_path);
        if (!in) {
            return;
        }
        std::string magic;
        int version = 0;
        if (!(in >> magic >> version) || magic != "moe-cache-profile" || version != 1) {
            LLAMA_LOG_WARN("%s: ignoring unknown MoE cache profile %s\n", __func__, profile_path.c_str());
            return;
        }
        size_t n_layers = 0;
        if (!(in >> n_layers) || n_layers != layers.size()) {
            LLAMA_LOG_WARN("%s: MoE cache profile %s does not match the model, ignoring it\n", __func__, profile_path.c_str());
            return;
        }
        profile.cached.resize(n_layers);
        for (size_t il = 0; il < n_layers; ++il) {
            int64_t layer_id = 0;
            int32_t n = 0;
            if (!(in >> layer_id >> n) || layer_id != (int64_t) il || n < 0 || n > (int32_t) layers[il].n_expert) {
                LLAMA_LOG_WARN("%s: MoE cache profile %s is corrupt, ignoring it\n", __func__, profile_path.c_str());
                profile.cached.clear();
                return;
            }
            profile.cached[il].resize(n);
            for (int32_t i = 0; i < n; ++i) {
                in >> profile.cached[il][i];
            }
        }
        profile.loaded = true;
        LLAMA_LOG_INFO("%s: seeded the placement from the MoE cache profile %s\n", __func__, profile_path.c_str());
    }

    void save_profile() const {
        if (profile_path.empty()) {
            return;
        }
        std::ofstream out(profile_path, std::ios::trunc);
        if (!out) {
            return;
        }
        out << "moe-cache-profile 1\n";
        out << layers.size() << "\n";
        for (size_t il = 0; il < layers.size(); ++il) {
            const layer & l = layers[il];
            std::vector<int32_t> ids;
            for (int32_t e = 0; e < l.n_expert; ++e) {
                if (l.slot_of[e] >= 0) {
                    ids.push_back(e);
                }
            }
            out << il << " " << ids.size();
            for (int32_t e : ids) {
                out << " " << e;
            }
            out << "\n";
        }
    }

    void set_initial(layer & l, int32_t il) {
        l.slot_of.assign(l.n_expert, -1);
        l.host_slot_of.assign(l.n_expert, -1);

        // the profile selects the experts that were cached at the end of the previous run
        std::vector<int32_t> cached_ids;
        if (profile_layer(il, l)) {
            cached_ids = profile.cached[il];
        } else {
            for (int32_t e = 0; e < l.n_cached_experts_per_layer; ++e) {
                cached_ids.push_back(e);
            }
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
            return usage[il][e].load(std::memory_order_relaxed);
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
        // incumbent keeps its row and the promotion waits for a later rebalance
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

        // give the freed bank rows to the promoted experts
        std::vector<int32_t> new_slot_of(n, -1);
        std::vector<bool> used(n_cached, false);
        for (int32_t e : final_sel) {
            const int32_t s = l.slot_of[e];
            if (s >= 0) {
                new_slot_of[e] = s;
                used[s]        = true;
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
        for (size_t i = 0; i < new_experts.size(); ++i) {
            new_slot_of[new_experts[i]] = free_slots[i];
        }
        budget -= new_experts.size();

        // the host banks of the cache keep the experts that are not on the device
        // the promoted experts free their host slots, the evicted experts take them
        std::vector<int32_t> new_host_slot_of;
        if (l.host_banked) {
            std::vector<int32_t> freed_host_slots;
            for (int32_t e = 0; e < n; ++e) {
                if (l.slot_of[e] < 0 && new_slot_of[e] >= 0) {
                    freed_host_slots.push_back(l.host_slot_of[e]);
                }
            }
            new_host_slot_of.assign(n, -1);
            size_t k = 0;
            for (int32_t e = 0; e < n; ++e) {
                if (new_slot_of[e] >= 0) {
                    continue;
                }
                if (l.host_slot_of[e] >= 0) {
                    new_host_slot_of[e] = l.host_slot_of[e];
                } else {
                    GGML_ASSERT(k < freed_host_slots.size());
                    new_host_slot_of[e] = freed_host_slots[k++];
                    // the promoted expert no longer needs its host copy, reuse the slot for the evicted one
                    add_host_fill(l, (int32_t) il, e, new_host_slot_of[e], 1);
                }
            }
            GGML_ASSERT(k == freed_host_slots.size());
        }

        // keep the details for the idle log, the usage counters are reset below
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
        st.n_selections = l.usage_total;
        for (int32_t i = 0; i < std::min<int32_t>(10, n_cached); ++i) {
            st.top.emplace_back(order[i], usage[il][order[i]].load(std::memory_order_relaxed));
        }
        last_rebalance_stats.push_back(std::move(st));

        l.slot_of.swap(new_slot_of);
        if (l.host_banked) {
            l.host_slot_of.swap(new_host_slot_of);
        }
        write_maps(l);

        // queue the uploads of the promoted experts, merged when consecutive
        add_uploads_merged(l, (int32_t) il, new_experts);

        for (int32_t e = 0; e < n; ++e) {
            usage[il][e].store(0, std::memory_order_relaxed);
        }
        l.usage_total   = 0;
        l.n_rebalances += 1;
        n_rebalances_total += 1;
    }

    void count_ids(size_t il, const int32_t * ids, size_t n) {
        layer & l = layers[il];
        if (l.id < 0) {
            return;
        }
        const int32_t n_experts = l.n_expert;
        for (size_t i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e >= 0 && e < n_experts) {
                usage[il][e].fetch_add(1, std::memory_order_relaxed);
                if (l.slot_of[e] >= 0) {
                    l.n_hits += 1;
                } else {
                    l.n_misses += 1;
                }
            }
        }
        l.usage_total  += n;
        n_selected_total += n;
    }

    void finish_reads() {
        for (size_t il = 0; il < layers.size(); ++il) {
            layer & l = layers[il];
            if (!l.read_pending) {
                continue;
            }
            GGML_ASSERT(l.read_buf);
            count_ids(il, (const int32_t *) ggml_backend_buffer_get_base(l.read_buf.get()), l.read_nbytes/sizeof(int32_t));
            l.read_pending = false;
        }
    }

    // rebalance the layers that collected enough selections, or all of them when forced
    uint64_t rebalance_layers(bool force, uint64_t budget) {
        uint64_t moved = 0;
        for (size_t il = 0; il < layers.size() && budget > 0; ++il) {
            layer & l = layers[il];
            if (l.id < 0) {
                continue;
            }
            if (!force && l.usage_total < l.rebalance_min) {
                continue;
            }
            const uint64_t before = budget;
            rebalance_layer(il, budget);
            moved += before - budget;
        }
        return moved;
    }

    const layer_info * get_layer(int32_t il) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].id < 0) {
            return nullptr;
        }
        return &layers[il].info;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    ggml_tensor * get_host_experts(const ggml_tensor * w) const {
        const auto it = host_bindings.find(w);
        return it != host_bindings.end() ? it->second : nullptr;
    }

    // called before each graph compute: complete the reads of the selected experts and count them
    // no expert moves here, the counters are applied only when the context is idle, see rebalance()
    void begin() {
        bool dirty = false;
        for (device & d : devices) {
            if (d.dirty) {
                ggml_backend_synchronize(d.backend);
                dirty = true;
            }
        }
        if (dirty) {
            finish_reads();
            for (device & d : devices) {
                d.dirty = false;
            }
        }
    }

    // apply the usage counters while the devices are idle, e.g. after all server slots are released
    // when forced the per layer threshold is ignored, e.g. to apply the usage of a prompt before the decode
    void rebalance(bool force) {
        const uint64_t rebalances_before = n_rebalances_total;
        const uint64_t uploaded_before   = n_uploaded_total;
        const size_t   bytes_before      = uploaded_bytes_total;

        for (device & d : devices) {
            ggml_backend_synchronize(d.backend);
            d.dirty = false;
        }
        finish_reads();

        // only the layers that collected enough selections since the last rebalance move their experts
        last_rebalance_stats.clear();
        const int64_t t_start_us = ggml_time_us();
        const uint64_t moved = rebalance_layers(force, max_promote);
        run_uploads(std::move(pending_jobs));
        const double elapsed_s = (ggml_time_us() - t_start_us)/1e6;

        if (n_rebalances_total > rebalances_before) {
            uint64_t n_loaded_experts = 0;
            uint64_t n_selections = 0;
            for (const auto & st : last_rebalance_stats) {
                n_loaded_experts += (uint64_t) st.n_loaded;
                n_selections     += st.n_selections;
            }
            LLAMA_LOG_INFO("llama_moe_cache: rebalance%s: %" PRIu64 " layer(s), %" PRIu64 " experts promoted, "
                    "%" PRIu64 " tensor uploads (%.2f MiB, %.2f MiB/s), %" PRIu64 " selections counted, %.2f s\n",
                force ? " (forced)" : " (idle)",
                n_rebalances_total - rebalances_before, n_loaded_experts,
                n_uploaded_total - uploaded_before, (uploaded_bytes_total - bytes_before)/1024.0/1024.0,
                (uploaded_bytes_total - bytes_before)/1024.0/1024.0/std::max(1e-6, elapsed_s),
                n_selections, elapsed_s);
            for (const auto & st : last_rebalance_stats) {
                std::string top;
                char buf[32];
                for (const auto & t : st.top) {
                    snprintf(buf, sizeof(buf), top.empty() ? "%d(%u)" : " %d(%u)", (int) t.first, (unsigned) t.second);
                    top += buf;
                }
                LLAMA_LOG_INFO("llama_moe_cache:   layer %2d: unloaded %2d, loaded %2d, kept %2d, top remaining ids: %s\n",
                    (int) st.il, (int) st.n_unloaded, (int) st.n_loaded, (int) st.n_kept, top.c_str());
            }
        }
        if (moved > 0) {
            save_profile();
        }
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        layer & l = layers[il];
        device & d = devices[l.id];

        GGML_ASSERT(backend == d.backend);

        // the get_rows that looks up the bank rows of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight
        const ggml_tensor * sel = lookup->src[1];
        GGML_ASSERT(ggml_is_contiguous(sel));

        d.dirty = true;

        // read the selected experts for the usage counters without synchronizing the device
        const size_t nbytes = ggml_nbytes(sel);
        ggml_backend_buffer_type_t buft_host = ggml_backend_dev_host_buffer_type(ggml_backend_get_device(backend));
        if (buft_host != nullptr) {
            if (l.read_buf_size < nbytes) {
                l.read_buf.reset(ggml_backend_buft_alloc_buffer(buft_host, nbytes));
                if (!l.read_buf) {
                    throw std::runtime_error("failed to allocate the MoE cache read buffer");
                }
                l.read_buf_size = nbytes;
            }
            ggml_backend_tensor_get_async(backend, sel, ggml_backend_buffer_get_base(l.read_buf.get()), 0, nbytes);
            l.read_pending = true;
            l.read_nbytes  = nbytes;
        } else {
            l.read_data.resize(nbytes/sizeof(int32_t));
            ggml_backend_tensor_get(sel, l.read_data.data(), 0, nbytes);
            count_ids(il, l.read_data.data(), l.read_data.size());
        }

        // copy the slot map, the next copy synchronizes the device before it changes again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        if (n_selected_total == 0) {
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

llama_moe_cache::llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
    pimpl(new impl(model, no_alloc, backends, bufts, size)) {
}

llama_moe_cache::~llama_moe_cache() = default;

const llama_moe_cache::layer_info * llama_moe_cache::get_layer(int32_t il) const {
    return pimpl->get_layer(il);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

ggml_tensor * llama_moe_cache::get_host_experts(const ggml_tensor * w) const {
    return pimpl->get_host_experts(w);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

void llama_moe_cache::begin() {
    pimpl->begin();
}

void llama_moe_cache::rebalance() {
    pimpl->rebalance(true);
}

void llama_moe_cache::rebalance_idle() {
    pimpl->rebalance(false);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    for (const auto & [buft, b] : pimpl->host_banks) {
        if (b.buf) {
            res[buft] += b.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    return res;
}
