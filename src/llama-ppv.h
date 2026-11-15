#pragma once

// Position-parallel verification (PPV), experimental.
//
// W ranks hold the same model and the same KV cache. In a PPV decode every rank submits the same
// ubatch of W tokens (one draft chain), but rank r computes only token r: its row of the embeddings,
// its Q/K/V, its attention row and its FFN. After the K/V projection of each layer the ranks
// all-gather the new tokens' K/V over a TCP full mesh, and each rank writes all W rows into its cache,
// so token r attends exactly to tokens < r. Each rank outputs the logits of its own token.
//
// env: LLAMA_PPV_PEERS=host:port,...  (rank order)   LLAMA_PPV_RANK=r

#include "llama.h"

struct ggml_context;
struct ggml_tensor;

// the public API (init, rank/world, set_row, allgather, stats) is declared in llama.h

// graph op: k [d, h, 1], v [d, h, 1] of this rank's token -> f32 [d*h*2, world] (k then v, rank order)
ggml_tensor * llama_ppv_gather_kv(ggml_context * ctx, ggml_tensor * k, ggml_tensor * v);
