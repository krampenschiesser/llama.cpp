#pragma once

#include "ggml-backend.h"

#include "llama.h"

#include <map>
#include <memory>
#include <vector>

struct ggml_cgraph;

struct llama_model;

// device cache for the experts of the MoE layers that keep some of their experts in host memory
//
// each cached layer has a fixed set of experts in device memory and a slot map on the device. the
// experts that are not cached are computed on the CPU from the host experts, with the CPU map.
// during inference no experts are copied between the device and the host.
//
// a layer whose whole expert set fits in the budget is cached in full: the device holds every expert
// and the graph computes it as a plain FFN with the expert ids, so there is no slot map, no CPU part
// and no usage buffer for that layer.
//
// the graph of a cached layer reads the experts of the cache at the slots of the selected experts
// and the misses at the zero rows at the end of the device banks, so that the device computes
// them as zero, and the CPU part computes the misses from the host experts. the two results are
// added.
//
// the slot map is updated only when the placement changes, i.e. while the context is idle. the
// selected experts are recorded in a device buffer of the layer and read back after the graph
// compute, so the placement does not add copies to the inference path.
//
// when the model has no host copy of the experts (llama_model::is_moe_expert_deferred), the cache
// owns the host banks too and keeps there only the experts that are not in the device cache. the
// host banks are filled from the plain GGUF source.
//
// the selected experts are counted on the host after the graph compute. the placement ranks the
// experts by a long history that forgets 1/8 of its counts per rebalance, while the rebalance
// trigger counts only the selections since the last rebalance. when the devices are idle and enough
// experts were counted, the most used experts are copied to the device cache and the maps are
// updated. the two rows of a swap are exchanged between the banks, the model file is read only when
// a type has no derepack or to fill the host banks when they are created.
//
// environment:
//   LLAMA_MOE_CACHE_RESERVE_MIB    free device memory kept for the KV cache and the compute buffers
//   LLAMA_MOE_CACHE_REBALANCE      selections per layer before an idle rebalance
//   LLAMA_MOE_CACHE_HYSTERESIS     factor a promoted expert must beat the evicted one by
//   LLAMA_MOE_CACHE_MAX_PROMOTE    max experts promoted by one rebalance
//   LLAMA_MOE_CACHE_MAX_PROMOTE_MIB max bytes promoted by one rebalance
//   LLAMA_MOE_CACHE_MAX_PROMOTE_MS target duration of a forced rebalance
//   LLAMA_MOE_CACHE_OVERLAP        run the CPU misses while the device computes (default off)
//   LLAMA_MOE_CACHE_TRACE          log the per graph cost of the bookkeeping (default off)
//   LLAMA_MOE_CACHE_CPU_DEVICE     allow the CPU backend as the cache device (tests only)
class llama_moe_cache {
public:
    struct layer_info {
        ggml_backend_t backend;  // the device that caches the layer
        ggml_tensor  * slot_map; // I32 [1, n_expert] in device memory: slot + 1 for the cached experts, 0 for the others
        ggml_tensor  * cpu_map;  // I32 [1, n_expert] in host memory: host slot for the experts read on the CPU, -1 for the cached ones
        ggml_tensor  * k_base;   // F32 [n_expert_used_max] in device memory: expert position of the slot expression
        int32_t        n_cached_experts_per_layer; // number of experts of the layer kept in the device cache
        int32_t        n_zero;   // zero rows at the end of the device banks, the misses read them
        bool           skip_negative; // the device skips the misses by id, the banks have no zero rows
        int32_t        n_host;   // host slots of the layer, n_expert if the model keeps the host copy
        int32_t        n_expert; // experts of the layer

        // ggml_get_rows(slot_map, selected_experts) returns the slot + 1 of the cached experts and 0
        // for the misses, the graph must replace the 0 with n_cached_experts_per_layer + the expert position of the token
    };

    // backends are all the backends of the context and n_ubatch is the max tokens of a batch
    llama_moe_cache(const llama_model & model, bool no_alloc, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size, int32_t n_ubatch);
    ~llama_moe_cache();

    // the cache of layer il, nullptr if the layer is not cached or is cached in full
    const layer_info * get_layer(int32_t il) const;

    // the device storage of the experts of w, n_cached_experts_per_layer + n_zero + 1 rows, nullptr if
    // w is not cached. reads the slots of a partially cached layer, see get_full_experts
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // the device storage of all the experts of w, nullptr unless the layer is cached in full. the ids
    // are the expert ids, the graph reads it as a plain device FFN
    ggml_tensor * get_full_experts(const ggml_tensor * w) const;

    // the host storage of the experts of w owned by the cache, nullptr if the model keeps the host copy
    ggml_tensor * get_host_experts(const ggml_tensor * w) const;

    // the device buffer that receives the selected experts of layer il for this graph, nullptr if
    // they do not fit. the graph must copy the ids into it, see end()
    ggml_tensor * get_usage(ggml_context * ctx, ggml_cgraph * gf, int32_t il, int64_t n_used, int64_t n_tokens) const;

    // called before a graph is built, the selected rows of the next graph replace the ones of the
    // last build. a reused graph does not call it
    void new_graph();

    // called after the graph compute is queued: reads back the selected experts of this graph
    void end(ggml_cgraph * gf);

    // applies the selections that are still in flight, e.g. before the stats or a rebalance
    void finish();

    // move the most used experts, the devices must be idle, call only when no work is in flight
    // rebalance() ignores the per layer threshold, rebalance_idle() applies it (see the env vars above)
    void rebalance();
    void rebalance_idle();

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

    // aggregate usage counters and memory of the cache
    llama_moe_cache_stats stats() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
