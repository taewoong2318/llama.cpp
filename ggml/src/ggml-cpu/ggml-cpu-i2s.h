// BitNet I2_S GEMV/GEMM declarations
#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

// nr activation columns (intended for nr < 4; the nr >= 4 case is better served by
// ggml_gemm_i2_i8_s below), nc weight rows, n inner dim. Output layout matches
// ggml_gemm_i2_i8_s: s[col * bs + row].
void ggml_gemv_i2_i8_s(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);
// nr activation columns, nc weight rows, n inner dim. s[col * bs + row].
void ggml_gemm_i2_i8_s(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);

// Interleaved (8-row) variants — operate on repacked weight data
void ggml_gemv_i2_i8_s_interleaved(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx_base, int64_t row_start, int64_t a_row_bytes,
    const void * GGML_RESTRICT vy, int nr, int nc);
void ggml_gemm_i2_i8_s_interleaved(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx_base, int64_t row_start, int64_t a_row_bytes,
    const void * GGML_RESTRICT vy, int nr, int nc);

// Exact union-sparse kernels (NEON only; ggml_i2s_sparse_available() reports support).
// Row-sparse: rows[0..na) of the weight, s[col * bs + i] for the i-th listed row.
void ggml_gemm_i2_i8_s_rows(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx,
                            const int32_t * rows, int na, const void * GGML_RESTRICT vy, int nr);
// K-major 2-bit copy of an I2_S weight (32-row blocks), built once per weight and cached.
const uint8_t * ggml_i2s_kmajor32_get(const void * vx, int n, int nrows);
// Column-sparse: row blocks [rb0, rb1) of 32 rows, active inner columns cols[0..na),
// X = active int8 activations widened to int16, [na][nr]. s[col * bs + row - rb0*32] = sum w*x.
void ggml_gemm_i2_kmajor32(int n, float * GGML_RESTRICT s, size_t bs, const uint8_t * GGML_RESTRICT wk,
                           int rb0, int rb1, const int32_t * cols, int na, const int16_t * X, int nr);
bool ggml_i2s_sparse_available(void);

#ifdef __cplusplus
}
#endif
