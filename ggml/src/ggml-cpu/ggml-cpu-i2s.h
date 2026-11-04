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

#ifdef __cplusplus
}
#endif
