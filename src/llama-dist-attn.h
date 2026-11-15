#pragma once

#include "ggml.h"

// Experimental: single-token decode attention over a KV cache that is sharded by position across
// processes. Every rank runs the full model; per layer, each rank attends over the cells it owns and
// the partial (output, max, sum) per head are all-gathered and merged with log-sum-exp.
//
// Enabled by env vars:
//   LLAMA_DIST_PEERS = host:port,host:port,...  (one per rank)
//   LLAMA_DIST_RANK  = this rank
//   LLAMA_DIST_OWN   = lo:hi  KV cells owned by this rank (default 0:inf)
// LLAMA_DIST_PEERS with a single entry runs the same kernels without networking.

bool llama_dist_attn_enabled();

// q: [hd, n_head, 1] f32, k/v: KV cache views [hd, n_head_kv, n_kv] f16, kq_mask: [n_kv, >=1] f16/f32
// returns [hd*n_head, 1] f32
ggml_tensor * llama_dist_attn(ggml_context * ctx, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v,
                              ggml_tensor * kq_mask, float kq_scale);
