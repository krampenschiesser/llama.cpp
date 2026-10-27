#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
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

static uint64_t llama_moe_cache_min_rebalance() {
    const char * env = getenv("LLAMA_MOE_CACHE_REBALANCE");
    if (env != nullptr && *env != '\0') {
        const long long value = atoll(env);
        return value > 0 ? (uint64_t) value : 1;
    }
    return 0;
}

}

struct llama_moe_cache::impl {
    struct device {
        ggml_backend_t backend = nullptr;
        ggml_backend_buffer_type_t buft = nullptr;
        size_t host_bytes = 0; // bytes of the host experts of the layers it caches
        double split = 0.0;    // share of the budget

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
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    const llama_model & model;
    size_t size;                  // device budget
    size_t read_alignment = 1;    // alignment of the plain reads
    uint64_t rebalance_min_env = 0;

    std::vector<device> devices;
    std::vector<layer> layers;

    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer
    std::unordered_map<const ggml_tensor *, ggml_tensor *> host_bindings; // model experts -> host bank

    // host banks, grouped by buffer type
    std::map<ggml_backend_buffer_type_t, host_bank> host_banks;

    // usage counters, one array per layer, incremented without synchronization
    std::vector<std::unique_ptr<std::atomic<uint32_t>[]>> usage;

    // staging buffers for the plain reads and the uploads
    std::vector<uint8_t> read_stage;
    std::vector<uint8_t> upload_stage;
    std::vector<uint8_t> host_stage;

    // maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    uint64_t n_selected_total  = 0;
    uint64_t n_uploaded_total  = 0;
    size_t uploaded_bytes_total = 0;
    size_t host_filled_bytes_total = 0;
    uint64_t n_rebalances_total = 0;

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

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
            model(model), size(size), layers(model.layers.size()) {
        read_alignment     = std::max<size_t>(1, model.plain_source_alignment());
        rebalance_min_env  = llama_moe_cache_min_rebalance();

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
        auto alloc_size = [](const layer & l, const device & d, int32_t n_cached_experts_per_layer) {
            const size_t alignment = ggml_backend_buft_get_alignment(d.buft);
            size_t res = 0;
            for (const ggml_tensor * t : l.experts) {
                res += GGML_PAD((size_t) t->nb[2]*(n_cached_experts_per_layer + l.n_zero + 1), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts like the layers, by the tensor split or by default by free memory
        const float * tensor_split = model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
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
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*l.host_bytes/d.host_bytes);
            // keep at least 2 experts on the host: the CPU part needs a 3D weight, so the host bank needs 2 planes
            const int32_t n_cached_max = l.n_expert - 2;
            while (l.n_cached_experts_per_layer < n_cached_max && alloc_size(l, d, l.n_cached_experts_per_layer + 1) <= budget) {
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
                n_tensors_host_bank[ggml_backend_buffer_get_type(l.experts[0]->buffer)] += (int) l.experts.size();
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
                    GGML_ASSERT(bank->nb[2] == t->nb[2]);
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

                l.rebalance_min = rebalance_min_env > 0 ? rebalance_min_env :
                    std::max<uint64_t>(1024, 4ull*l.n_expert);

                devices[l.id].buf_size += alloc_size(l, d, l.n_cached_experts_per_layer);
            }

            if (l.host_banked && l.n_host > 0) {
                const ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(l.experts[0]->buffer);
                host_bank & b = host_banks.at(buft);
                for (const ggml_tensor * t : l.experts) {
                    // mul_mat_id needs a 3D weight, so the host bank holds at least 2 planes
                    const int32_t n_planes = std::max<int32_t>(l.n_host, 2);
                    ggml_tensor * bank = ggml_new_tensor_3d(b.ctx.get(), t->type, t->ne[0], t->ne[1], n_planes);
                    GGML_ASSERT(bank->nb[2] == t->nb[2]);
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

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
                d.buf_size = ggml_backend_alloc_ctx_tensors_from_buft_size(d.ctx.get(), d.buft);
            }
            if (ctx_host) {
                buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                    t->buffer = buf_host.get();
                }
                buf_host_size = ggml_backend_alloc_ctx_tensors_from_buft_size(ctx_host.get(), buft_host);
            }
            for (auto & [buft, b] : host_banks) {
                b.buf.reset(ggml_backend_buft_alloc_buffer(buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(b.ctx.get()); t != nullptr; t = ggml_get_next_tensor(b.ctx.get(), t)) {
                    t->buffer = b.buf.get();
                }
                b.buf_size = ggml_backend_alloc_ctx_tensors_from_buft_size(b.ctx.get(), buft);
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            if (ctx_host) {
                buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
                if (!buf_host) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                buf_host_size = ggml_backend_buffer_get_size(buf_host.get());
            }
            for (auto & [buft, b] : host_banks) {
                b.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(b.ctx.get(), buft));
                if (!b.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache host buffers");
                }
                b.buf_size = ggml_backend_buffer_get_size(b.buf.get());
            }

            // the initial placement keeps the first experts of each layer on the device and the rest on the host
            for (layer & l : layers) {
                if (!l.experts.empty()) {
                    set_initial(l);
                }
            }
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

    bool read_experts_plain(const layer & l, size_t ip, int32_t src_expert, size_t n, void * dst) {
        const ggml_tensor * w = l.experts[ip];
        const size_t expert_size = w->nb[2];
        const size_t offs = (size_t) src_expert*expert_size;
        const size_t size = n*expert_size;
        if (!l.host_banked && ggml_backend_buffer_is_host(w->buffer)) {
            memcpy(dst, (const uint8_t *) w->data + offs, size);
            return true;
        }
        if (read_alignment <= 1) {
            return model.read_plain_source(w, offs, size, dst, nullptr, 0);
        }
        read_stage.resize(size + 3*read_alignment);
        uint8_t * stage = llama_moe_cache_align_ptr(read_stage.data(), read_alignment);
        const size_t stage_len = read_stage.size() - (size_t) (stage - read_stage.data());
        return model.read_plain_source(w, offs, size, dst, stage, stage_len);
    }

    // copy n consecutive experts of w to the device rows starting at dst_slot
    void upload_experts(layer & l, int32_t ip, int32_t src_expert, int32_t dst_slot, size_t n) {
        const ggml_tensor * w = l.experts[ip];
        const size_t expert_size = w->nb[2];
        upload_stage.resize(n*expert_size);
        if (!read_experts_plain(l, ip, src_expert, n, upload_stage.data())) {
            GGML_ABORT("the MoE cache cannot read the plain source of expert tensor %s", w->name);
        }
        ggml_backend_tensor_set(l.banks[ip], upload_stage.data(), (size_t) dst_slot*expert_size, n*expert_size);

        l.n_promoted        += n;
        l.promoted_bytes    += n*expert_size;
        n_uploaded_total    += n;
        uploaded_bytes_total += n*expert_size;
    }

    // copy n consecutive experts of w to the host bank rows starting at dst_slot
    void fill_host_experts(layer & l, int32_t src_expert, int32_t dst_slot, int32_t n) {
        if (!l.host_banked) {
            return;
        }
        // read and repack a few experts at a time to keep the staging buffer small
        const size_t expert_size = l.experts[0]->nb[2];
        const int32_t per_chunk = (int32_t) std::max<size_t>(1, 32u*1024*1024/expert_size);
        for (int32_t i = 0; i < n; i += per_chunk) {
            const int32_t k = std::min(per_chunk, n - i);
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                const size_t esz = l.experts[ip]->nb[2];
                host_stage.resize((size_t) k*esz);
                if (!read_experts_plain(l, ip, src_expert + i, k, host_stage.data())) {
                    GGML_ABORT("the MoE cache cannot read the plain source of expert tensor %s", l.experts[ip]->name);
                }
                ggml_backend_tensor_set(l.host_banks[ip], host_stage.data(), (size_t) (dst_slot + i)*esz, (size_t) k*esz);
                host_filled_bytes_total += (size_t) k*esz;
            }
        }
    }

    void set_initial(layer & l) {
        l.slot_of.assign(l.n_expert, -1);
        l.host_slot_of.assign(l.n_expert, -1);

        if (l.host_banked) {
            // the first experts of the layer go to the device, the others stay in the host banks
            for (int32_t e = l.n_cached_experts_per_layer; e < l.n_expert; ++e) {
                l.host_slot_of[e] = e - l.n_cached_experts_per_layer;
            }
        }
        if (l.id < 0) {
            // all experts are read on the CPU, the host banks are the only copy
            if (l.host_banked) {
                fill_host_experts(l, 0, 0, l.n_expert);
            }
            return;
        }
        for (int32_t e = 0; e < l.n_cached_experts_per_layer; ++e) {
            l.slot_of[e] = e;
        }
        write_maps(l);
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            upload_experts(l, (int32_t) ip, 0, 0, l.n_cached_experts_per_layer);
        }
        if (l.host_banked) {
            fill_host_experts(l, l.n_cached_experts_per_layer, 0, l.n_host);
        }
    }

    // the most used experts of the layer replace the least used ones
    void rebalance_layer(size_t il) {
        layer & l = layers[il];
        if (l.id < 0) {
            return;
        }
        const int32_t  n = l.n_expert;
        const int32_t  n_cached_experts_per_layer = l.n_cached_experts_per_layer;

        // select the experts with the most selections, retained experts win the ties
        std::vector<int32_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        auto count = [&](int32_t e) {
            return usage[il][e].load(std::memory_order_relaxed);
        };
        std::partial_sort(order.begin(), order.begin() + n_cached_experts_per_layer, order.end(), [&](int32_t a, int32_t b) {
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

        // keep the bank rows of the retained experts, give the free rows to the new experts
        std::vector<int32_t> new_slot_of(n, -1);
        std::vector<bool> used(n_cached_experts_per_layer, false);
        for (int32_t i = 0; i < n_cached_experts_per_layer; ++i) {
            const int32_t e  = order[i];
            const int32_t s  = l.slot_of[e];
            if (s >= 0) {
                new_slot_of[e] = s;
                used[s]        = true;
            }
        }
        std::vector<int32_t> free_slots;
        for (int32_t s = 0; s < n_cached_experts_per_layer; ++s) {
            if (!used[s]) {
                free_slots.push_back(s);
            }
        }
        std::vector<int32_t> new_experts;
        for (int32_t i = 0; i < n_cached_experts_per_layer; ++i) {
            const int32_t e = order[i];
            if (new_slot_of[e] < 0) {
                new_experts.push_back(e);
            }
        }
        GGML_ASSERT(new_experts.size() == free_slots.size());
        std::sort(new_experts.begin(), new_experts.end());
        for (size_t i = 0; i < new_experts.size(); ++i) {
            new_slot_of[new_experts[i]] = free_slots[i];
        }

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
                    fill_host_experts(l, e, new_host_slot_of[e], 1);
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
        for (int32_t i = 0; i < std::min<int32_t>(10, n_cached_experts_per_layer); ++i) {
            const int32_t e = order[i];
            st.top.emplace_back(e, usage[il][e].load(std::memory_order_relaxed));
        }
        last_rebalance_stats.push_back(std::move(st));

        // upload the new experts, merged when consecutive experts go to consecutive bank rows
        for (size_t i = 0; i < new_experts.size(); ) {
            size_t k = 1;
            while (i + k < new_experts.size() &&
                    new_experts[i + k] == new_experts[i] + (int32_t) k &&
                    new_slot_of[new_experts[i + k]] == new_slot_of[new_experts[i]] + (int32_t) k) {
                k++;
            }
            for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                upload_experts(l, (int32_t) ip, new_experts[i], new_slot_of[new_experts[i]], k);
            }
            i += k;
        }

        l.slot_of.swap(new_slot_of);
        if (l.host_banked) {
            l.host_slot_of.swap(new_host_slot_of);
        }
        write_maps(l);

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

    void maybe_rebalance() {
        for (size_t il = 0; il < layers.size(); ++il) {
            layer & l = layers[il];
            if (l.id >= 0 && l.usage_total >= l.rebalance_min) {
                rebalance_layer(il);
            }
        }
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
    void rebalance() {
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
        maybe_rebalance();

        if (n_rebalances_total > rebalances_before) {
            uint64_t n_loaded_experts = 0;
            uint64_t n_selections = 0;
            for (const auto & st : last_rebalance_stats) {
                n_loaded_experts += (uint64_t) st.n_loaded;
                n_selections     += st.n_selections;
            }
            LLAMA_LOG_INFO("llama_moe_cache: idle rebalance: %" PRIu64 " layer(s), %" PRIu64 " experts promoted, "
                    "%" PRIu64 " tensor uploads (%.2f MiB), %" PRIu64 " selections counted\n",
                n_rebalances_total - rebalances_before, n_loaded_experts,
                n_uploaded_total - uploaded_before, (uploaded_bytes_total - bytes_before)/1024.0/1024.0,
                n_selections);
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
        LLAMA_LOG_INFO("llama_moe_cache: counted %" PRIu64 " selected experts, %" PRIu64 " rebalances, %" PRIu64 " experts uploaded (%.2f MiB)\n",
            n_selected_total, n_rebalances_total, n_uploaded_total, uploaded_bytes_total/1024.0/1024.0);
        for (size_t il = 0; il < layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.id < 0 || l.n_rebalances == 0) {
                continue;
            }
            LLAMA_LOG_INFO("llama_moe_cache: layer %2d: %d cached experts, %" PRIu64 " rebalances, %" PRIu64 " experts promoted (%.2f MiB)\n",
                (int) il, l.n_cached_experts_per_layer, l.n_rebalances, l.n_promoted, l.promoted_bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size) :
    pimpl(new impl(model, backends, bufts, size)) {
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
    pimpl->rebalance();
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
