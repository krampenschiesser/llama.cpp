#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <vector>

struct llama_model;

// device cache for the experts of the MoE layers that keep some of their experts in host memory
//
// each cached layer has a fixed set of experts in device memory and a slot map in host memory.
// the experts that are not cached are computed on the CPU from the host experts, with the CPU map.
// during inference no experts are copied between the device and the host.
//
// the graph of a cached layer reads the experts of the cache at the slots of the selected experts
// and the misses at the zero rows at the end of the device banks, so that the device computes
// them as zero, and the CPU part computes the misses from the host experts. the two results are
// added.
//
// when the model has no host copy of the experts (llama_model::is_moe_expert_deferred), the cache
// owns the host banks too and keeps there only the experts that are not in the device cache. the
// host banks are filled from the plain GGUF source.
//
// the selected experts are counted on the host with the reads done by the slot map copies. when
// the devices are idle and enough experts were counted, the most used experts are copied to the
// device cache from the plain source of the model file and the maps are updated. the experts that
// leave the device cache are written back to the host banks.
class llama_moe_cache {
public:
    struct layer_info {
        ggml_backend_t backend;  // the device that caches the layer
        ggml_tensor  * slot_map; // I32 [1, n_expert] in host memory: slot + 1 for the cached experts, 0 for the others
        ggml_tensor  * cpu_map;  // I32 [1, n_expert] in host memory: host slot for the experts read on the CPU, -1 for the cached ones
        int32_t        n_cached_experts_per_layer; // number of experts of the layer kept in the device cache
        int32_t        n_zero;   // zero rows at the end of the device banks, the misses read them
        int32_t        n_host;   // host slots of the layer, n_expert if the model keeps the host copy
        int32_t        n_expert; // experts of the layer

        // ggml_get_rows(slot_map, selected_experts) returns the slot + 1 of the cached experts and 0
        // for the misses, the graph must replace the 0 with n_cached_experts_per_layer + the expert position of the token
    };

    // backends are all the backends of the context, each device caches the MoE layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size);
    ~llama_moe_cache();

    // the cache of layer il, nullptr if the layer is not cached
    const layer_info * get_layer(int32_t il) const;

    // the device storage of the experts of w, n_cached_experts_per_layer + n_zero + 1 rows, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the host storage of the experts of w owned by the cache, nullptr if the model keeps the host copy
    ggml_tensor * get_host_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback: copies the slot map of a cached layer to the device and
    // records the selected experts of the split for the usage counters, returns false if src is
    // not the slot map of a cached layer
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // called before each graph compute: completes the reads of the selected experts and counts them
    // no expert moves here, the counters are applied only by rebalance() while the context is idle
    void begin();

    // move the most used experts to the layers that collected enough selections, the devices must be idle
    // call only when no work is in flight, e.g. when all server slots are released
    void rebalance();

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
