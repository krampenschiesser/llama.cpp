# MoE cache split pipeline

The MoE cache (`--n-cpu-moe` with `--moe-cache-mib`) keeps part of the experts of a layer on the
device and the rest in host memory. A cached layer is computed by two halves: the device computes
the cached experts and the CPU computes the misses from the host bank. This document describes how
the two halves are scheduled so that they run at the same time.

## Splits of a cached layer

Each cached layer is split into four scheduler splits:

```
split 1  device    gate matmul, topk, weights, slot lookup, expert ids for the CPU half
split 2  host      cpu_ids lookup, up/gate/down mul_mat_id on the host bank, activation
split 3  device    mul_mat_id on the device bank, activation
split 4  device    merge add, down bias, post norm, router weights, per expert adds
```

`llama-graph.cpp` forces the merge add onto the layer device and marks it as a split barrier, so
split 3 is queued before split 2 is joined. The guest list of the scheduler is built in the order
above, so the copies of the host split inputs sit between the topk split and the device half split.

## Copies

Per cached layer and per graph:

| copy | size | mechanism |
|------|------|-----------|
| `ids` (selected experts) device to host | `n_used*n_tokens*4` | pinned host buffer, event |
| `cur` (FFN input) device to host | `n_embd*n_tokens*4` | pinned host buffer, event |
| `experts_cpu` host to device | `n_embd*n_used*n_tokens*4` | device stream, stream ordered |

The device to host copies run on a second CUDA stream (`ggml_backend_copy_d2h`, optional, see
`GGML_SCHED_NO_COPY_STREAM`). The host split waits for the copy event before it runs, the main
thread never waits for a copy. The source tensors of the copies are marked as outputs of the graph
so that the allocator does not reuse them before the copy has been consumed.

The host split runs on a worker thread (`ggml_backend_sched_set_cpu_overlap`). The worker is joined
before the first split that reads host memory, which is the merge add.

## Zero suppression

The two halves are added with a single `ggml_add`, each half is zero where the other one is
responsible:

- device misses read the zero rows at the end of the bank, or, when the device advertises
  `ggml_backend_mul_mat_id_skip_negative`, carry id `-1` that the kernel skips and zeroes;
- CPU hits carry id `-1` and the CPU `mul_mat_id` zeroes those columns.

## Configuration

```
LLAMA_MOE_CACHE_PIPE        bit mask, 1 = async split copies, 2 = host split on a worker
                            (default 3, 0 restores the synchronous inline path)
GGML_SCHED_NO_COPY_STREAM   keep the device to host copies on the compute stream (debug)
GGML_SCHED_DEBUG=2          print the split count and the copy counters of each compute
LLAMA_MOE_CACHE_TRACE=1     per graph bookkeeping cost of the cache
```

Copy counters of one decode graph of a 7 layer cache: `4 splits per layer, ~23 async copies,
0 sync copies`.

## Limits

The merge add feeds the attention of the same layer, so the pipeline depth is one layer: the gain
per layer is bounded by the smaller of the two halves. With a cache budget that leaves the CPU half
much larger than the device half, the placement (and not the scheduler) is the limiting factor.
