#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// work buffer bytes for src1 [ne10, ne11] in the token-tiled int8 layout + per-token scales
size_t ggml_vlut_wsize(int64_t ne10, int64_t ne11);
void   ggml_vlut_quantize_act(const char * src, size_t nb11, int64_t ne10, int64_t ne11, void * wdata, int ith, int nth);

// s[r*bs + c] = sum_k x[c,k] * y[r,k] for nc weight rows starting at vx, nr tokens
void ggml_gemm_i2v4_i8v_lut(int ith, int nth, int n, float * GGML_RESTRICT s, size_t bs,
                            const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc);

#ifdef __cplusplus
}
#endif
