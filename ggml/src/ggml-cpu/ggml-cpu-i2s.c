// BitNet I2_S GEMV/GEMM optimized kernels
// Extracted from ggml-aarch64.c

#include "ggml-quants.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "ggml.h"

#include <string.h>
#include <assert.h>
#include <math.h>

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#define QK_I2_S 128
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define QK_I2_S 64
#else
#define QK_I2_S 128
#endif

#include "gemm-config.h"

#if defined(__ARM_NEON)
// Shared NEON tile: processes 4 weight rows (A) x up to `rn` (<=4) activation columns
// (B) at once, unpacking each row's 2-bit codes once per 128-element block and
// reusing them across all `rn` columns -- instead of re-unpacking/re-reading the
// weight range once per column, which is what the scalar fallback below does. Same
// bit layout as ggml_vec_dot_i2_i8_s_1x1's NEON path (quants.c): per 128-element
// block, 2 halves of 16 packed bytes, 4 groups of 16 codes per half via
// shift-and-mask. Uses ggml_vdotq_s32 (ggml-cpu-impl.h), so this automatically gets
// the native SDOT instruction when __ARM_FEATURE_DOTPROD is available and the
// proven vmull_s8-based emulation otherwise.
//
// Used with rn=4 by ggml_gemm_i2_i8_s's main tile, rn<4 by its column remainder, and
// rn=nr (always <=3) by the whole of ggml_gemv_i2_i8_s -- one shared implementation
// of the correctness-critical unpacking instead of three near-duplicates of it.
static void i2s_neon_dot_tile4xN(
        const uint8_t * GGML_RESTRICT x_rows[4],
        const int8_t  * GGML_RESTRICT y_cols[4],
        int rn, int64_t n, float out[4][4]) {
    const int64_t nb = n / 128;
    const uint8x16_t mask2 = vdupq_n_u8(0x03);
    int32x4_t acc[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < rn; j++)
            acc[i][j] = vdupq_n_s32(0);

    for (int64_t bl = 0; bl < nb; bl++) {
        for (int half = 0; half < 2; half++) {
            int8x16_t a_vals[4][4]; // [row][group]
            for (int i = 0; i < 4; i++) {
                uint8x16_t packed = vld1q_u8(x_rows[i] + bl * 32 + half * 16);
                a_vals[i][0] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed, 6), mask2));
                a_vals[i][1] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed, 4), mask2));
                a_vals[i][2] = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(packed, 2), mask2));
                a_vals[i][3] = vreinterpretq_s8_u8(vandq_u8(packed, mask2));
            }
            for (int j = 0; j < rn; j++) {
                const int8_t * b_ptr = y_cols[j] + bl * 128 + half * 16;
                int8x16_t b0 = vld1q_s8(b_ptr + 0 * 32);
                int8x16_t b1 = vld1q_s8(b_ptr + 1 * 32);
                int8x16_t b2 = vld1q_s8(b_ptr + 2 * 32);
                int8x16_t b3 = vld1q_s8(b_ptr + 3 * 32);
                for (int i = 0; i < 4; i++) {
                    acc[i][j] = ggml_vdotq_s32(acc[i][j], a_vals[i][0], b0);
                    acc[i][j] = ggml_vdotq_s32(acc[i][j], a_vals[i][1], b1);
                    acc[i][j] = ggml_vdotq_s32(acc[i][j], a_vals[i][2], b2);
                    acc[i][j] = ggml_vdotq_s32(acc[i][j], a_vals[i][3], b3);
                }
            }
        }
    }

    int32_t sums[4][4];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < rn; j++)
            sums[i][j] = vaddvq_s32(acc[i][j]);

    // Scalar tail for n % 128 != 0 -- same byte/bit addressing as
    // ggml_vec_dot_i2_i8_s_1x1's own tail loop (quants.c). Never exercised by
    // BitNet-b1.58-2B-4T's own MLP shapes (both exact multiples of 128), but kept
    // for correctness on arbitrary n.
    for (int64_t idx = nb * 128; idx < n; idx++) {
        int64_t within   = idx - nb * 128;
        int64_t k        = within / 32;
        int64_t gp       = within % 32;
        int64_t byte_idx = nb * 32 + gp;
        int     bit_pos  = 6 - 2 * (int)k;
        for (int i = 0; i < 4; i++) {
            int code = (x_rows[i][byte_idx] >> bit_pos) & 0x03;
            for (int j = 0; j < rn; j++) {
                sums[i][j] += code * y_cols[j][idx];
            }
        }
    }

    for (int i = 0; i < 4; i++)
        for (int j = 0; j < rn; j++)
            out[i][j] = (float)sums[i][j];
}
#endif // __ARM_NEON

void ggml_gemv_i2_i8_s(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    // nr = number of activation columns (always < 4 here -- the >=4 case goes
    // through ggml_gemm_i2_i8_s instead), nc = number of weight rows, n = inner
    // dimension. Output layout matches ggml_gemm_i2_i8_s: s[col*bs+row].
#if defined(__ARM_NEON)
    // nr==1 has no cross-column reuse to offer (there is only one column), so route
    // it straight through ggml_vec_dot_i2_i8_s's own row loop (nrc=nc in one call)
    // instead of the 4-row tile below -- measured faster than paying the tile's
    // extra bookkeeping for zero benefit when there is nothing to reuse across.
    if (nr == 1) {
        ggml_vec_dot_i2_i8_s(n, s, 1, vx, n / 4, vy, 0, nc);
        return;
    }

    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t *)vy;
    const int64_t a_row_bytes = n / 4;

    int64_t r = 0;
    for (; r + 4 <= nc; r += 4) {
        const uint8_t * x_rows[4] = {
            x + (r + 0) * a_row_bytes, x + (r + 1) * a_row_bytes,
            x + (r + 2) * a_row_bytes, x + (r + 3) * a_row_bytes,
        };
        const int8_t * y_cols[4] = { NULL, NULL, NULL, NULL };
        for (int j = 0; j < nr; j++) {
            y_cols[j] = y + j * n;
        }

        float tile[4][4];
        i2s_neon_dot_tile4xN(x_rows, y_cols, nr, n, tile);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < nr; j++) {
                s[j * bs + (r + i)] = tile[i][j];
            }
        }
    }

    // Row remainder (<4 rows left): direct vec_dot per (row, col) pair -- not
    // tiled, same precedent as ggml_gemm_i2_i8_s's own row remainder below.
    for (; r < nc; r++) {
        for (int64_t c = 0; c < nr; c++) {
            ggml_vec_dot_i2_i8_s(n, s + c * bs + r, 1,
                (const uint8_t *)vx + r * a_row_bytes, n,
                (const int8_t *)vy + c * n, 0, 1);
        }
    }
#else
    // Generic fallback. Not performance-critical: on x86 this function is
    // unreachable in practice (llamafile_sgemm_i2s/tinyBLAS_I2S_AVX always wins
    // first and always succeeds whenever __AVX2__ is available), so this exists
    // purely for correctness in the (GGML_LLAMAFILE off, or non-AVX2 non-ARM)
    // configuration -- no SIMD work needed here.
    for (int64_t col = 0; col < nr; col++) {
        const uint8_t * vy_col = (const uint8_t *)vy + col * n;
        for (int64_t iir0 = 0; iir0 < nc; iir0++) {
            ggml_vec_dot_i2_i8_s(n, s + col * bs + iir0, 1, (const uint8_t *)vx + iir0 * n / 4, n, vy_col, 0, 1);
        }
    }
#endif
}

void ggml_gemm_i2_i8_s(int n, float * GGML_RESTRICT s, size_t bs, const void * GGML_RESTRICT vx, const void * GGML_RESTRICT vy, int nr, int nc) {
    // nr = number of activation columns (B side)
    // nc = number of weight rows (A side)
    // n  = inner dimension (elements per row)
    // vx = weight data (I2_S, 2-bit packed, nc rows, each n/4 bytes)
    // vy = activation data (I8_S, nr columns, each n bytes)
    // s  = output, s[col * bs + row], where col indexes B columns, row indexes A rows

#if defined(__AVX2__)
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t *)vy;
    const int64_t a_row_bytes = n / 4;
    const int64_t nb = n / 128;  // number of 128-element blocks

    const __m256i mask_2bit = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    // Process 4 weight rows (A) × 4 activation columns (B) at a time
    int64_t r = 0;
    for (; r + 4 <= nc; r += 4) {
        int64_t c = 0;
        for (; c + 4 <= nr; c += 4) {
            // 4×4 int32 accumulators
            __m256i acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    acc[i][j] = _mm256_setzero_si256();

            for (int64_t bl = 0; bl < nb; bl++) {
                // Load and unpack 4 rows of A (32 packed bytes → 4×32 uint8 values each)
                __m256i a_unpacked[4][4];
                for (int i = 0; i < 4; i++) {
                    const uint8_t * a_ptr = x + (r + i) * a_row_bytes + bl * 32;
                    __m256i packed = _mm256_loadu_si256((const __m256i *)a_ptr);
                    a_unpacked[i][0] = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask_2bit);
                    a_unpacked[i][1] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_2bit);
                    a_unpacked[i][2] = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask_2bit);
                    a_unpacked[i][3] = _mm256_and_si256(packed, mask_2bit);
                }

                // Load 4 columns of B (128 bytes each) and compute dot products
                for (int j = 0; j < 4; j++) {
                    const int8_t * b_ptr = y + (c + j) * n + bl * 128;
                    __m256i b0 = _mm256_loadu_si256((const __m256i *)(b_ptr));
                    __m256i b1 = _mm256_loadu_si256((const __m256i *)(b_ptr + 32));
                    __m256i b2 = _mm256_loadu_si256((const __m256i *)(b_ptr + 64));
                    __m256i b3 = _mm256_loadu_si256((const __m256i *)(b_ptr + 96));

                    // Each of 4 A rows multiplied with this B column
                    for (int i = 0; i < 4; i++) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][0], b0);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][1], b1);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][2], b2);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][3], b3);
#else
                        __m256i d0 = _mm256_maddubs_epi16(a_unpacked[i][0], b0);
                        __m256i d1 = _mm256_maddubs_epi16(a_unpacked[i][1], b1);
                        __m256i d2 = _mm256_maddubs_epi16(a_unpacked[i][2], b2);
                        __m256i d3 = _mm256_maddubs_epi16(a_unpacked[i][3], b3);
                        __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(d0, d1),
                                                         _mm256_add_epi16(d2, d3));
                        acc[i][j] = _mm256_add_epi32(acc[i][j],
                                                      _mm256_madd_epi16(sum16, one16));
#endif
                    }
                }
            }

            // Horizontal sum and store 4×4 results
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    __m128i sum128 = _mm_add_epi32(
                        _mm256_castsi256_si128(acc[i][j]),
                        _mm256_extractf128_si256(acc[i][j], 1));
                    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
                    __m128i sum64 = _mm_add_epi32(hi64, sum128);
                    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
                    s[(c + j) * bs + (r + i)] = (float)_mm_cvtsi128_si32(
                        _mm_add_epi32(sum64, hi32));
                }
            }
        }

        // Handle remaining B columns (< 4)
        for (; c < nr; c++) {
            __m256i acc_rem[4];
            for (int i = 0; i < 4; i++)
                acc_rem[i] = _mm256_setzero_si256();

            for (int64_t bl = 0; bl < nb; bl++) {
                const int8_t * b_ptr = y + c * n + bl * 128;
                __m256i b0 = _mm256_loadu_si256((const __m256i *)(b_ptr));
                __m256i b1 = _mm256_loadu_si256((const __m256i *)(b_ptr + 32));
                __m256i b2 = _mm256_loadu_si256((const __m256i *)(b_ptr + 64));
                __m256i b3 = _mm256_loadu_si256((const __m256i *)(b_ptr + 96));

                for (int i = 0; i < 4; i++) {
                    const uint8_t * a_ptr = x + (r + i) * a_row_bytes + bl * 32;
                    __m256i packed = _mm256_loadu_si256((const __m256i *)a_ptr);
                    __m256i a0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask_2bit);
                    __m256i a1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_2bit);
                    __m256i a2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask_2bit);
                    __m256i a3 = _mm256_and_si256(packed, mask_2bit);

#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a0, b0);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a1, b1);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a2, b2);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a3, b3);
#else
                    __m256i d0 = _mm256_maddubs_epi16(a0, b0);
                    __m256i d1 = _mm256_maddubs_epi16(a1, b1);
                    __m256i d2 = _mm256_maddubs_epi16(a2, b2);
                    __m256i d3 = _mm256_maddubs_epi16(a3, b3);
                    __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(d0, d1),
                                                     _mm256_add_epi16(d2, d3));
                    acc_rem[i] = _mm256_add_epi32(acc_rem[i],
                                                   _mm256_madd_epi16(sum16, one16));
#endif
                }
            }

            for (int i = 0; i < 4; i++) {
                __m128i sum128 = _mm_add_epi32(
                    _mm256_castsi256_si128(acc_rem[i]),
                    _mm256_extractf128_si256(acc_rem[i], 1));
                __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
                __m128i sum64 = _mm_add_epi32(hi64, sum128);
                __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
                s[c * bs + (r + i)] = (float)_mm_cvtsi128_si32(
                    _mm_add_epi32(sum64, hi32));
            }
        }
    }

    // Handle remaining A rows (< 4)
    for (; r < nc; r++) {
        for (int64_t c = 0; c < nr; c++) {
            ggml_vec_dot_i2_i8_s(n, s + c * bs + r, 1,
                (const uint8_t *)vx + r * a_row_bytes, n,
                (const int8_t *)vy + c * n, 0, 1);
        }
    }
#elif defined(__ARM_NEON)
    const uint8_t * x = (const uint8_t *)vx;
    const int8_t  * y = (const int8_t *)vy;
    const int64_t a_row_bytes = n / 4;

    // Process 4 weight rows (A) x 4 activation columns (B) at a time, via the
    // shared tile helper above.
    int64_t r = 0;
    for (; r + 4 <= nc; r += 4) {
        const uint8_t * x_rows[4] = {
            x + (r + 0) * a_row_bytes, x + (r + 1) * a_row_bytes,
            x + (r + 2) * a_row_bytes, x + (r + 3) * a_row_bytes,
        };

        int64_t c = 0;
        for (; c + 4 <= nr; c += 4) {
            const int8_t * y_cols[4] = {
                y + (c + 0) * n, y + (c + 1) * n, y + (c + 2) * n, y + (c + 3) * n,
            };
            float tile[4][4];
            i2s_neon_dot_tile4xN(x_rows, y_cols, 4, n, tile);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    s[(c + j) * bs + (r + i)] = tile[i][j];
                }
            }
        }

        // Column remainder (1-3 left): same tile helper, smaller rn, instead of
        // AVX2's separate hand-written remainder loop.
        if (c < nr) {
            const int rn = (int)(nr - c);
            const int8_t * y_cols[4] = { NULL, NULL, NULL, NULL };
            for (int j = 0; j < rn; j++) {
                y_cols[j] = y + (c + j) * n;
            }
            float tile[4][4];
            i2s_neon_dot_tile4xN(x_rows, y_cols, rn, n, tile);
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < rn; j++) {
                    s[(c + j) * bs + (r + i)] = tile[i][j];
                }
            }
        }
    }

    // Handle remaining A rows (< 4): direct vec_dot per (row, col) pair, same
    // precedent as the AVX2 path's own row remainder above.
    for (; r < nc; r++) {
        for (int64_t c = 0; c < nr; c++) {
            ggml_vec_dot_i2_i8_s(n, s + c * bs + r, 1,
                (const uint8_t *)vx + r * a_row_bytes, n,
                (const int8_t *)vy + c * n, 0, 1);
        }
    }
#else
    // Scalar / non-AVX2 fallback. ggml_vec_dot_i2_i8_s's nrc parameter already iterates
    // `nrc` WEIGHT ROWS against one FIXED activation vector (advancing vx by n/4 bytes
    // per row, writing results contiguously starting at s) -- it has no notion of
    // batching multiple activation columns against one fixed weight row. So each
    // activation column gets exactly one call covering all nc weight rows; there is no
    // additional row/column blocking to do on top of that.
    for (int64_t col = 0; col < nr; col++) {
        const void * vy_col = (const uint8_t *)vy + col * n;
        ggml_vec_dot_i2_i8_s(n, s + col * nc, 1, vx, n, vy_col, n, nc);
    }
#endif // __AVX2__ / __ARM_NEON
}

// =============================================================
// Interleaved (8-row repacked) GEMM/GEMV for I2_S
//
// Weight layout after repack: groups of 8 rows, within each group
// bytes are interleaved in 8-byte blocks:
//   [row0_b0..b7][row1_b0..b7]...[row7_b0..b7][row0_b8..b15]...
//
// vx_base points to the start of ALL weight data (interleaved)
// row_start is the first output row to process
// nc = number of output rows to process (must be multiple of 4)
// nr = number of B columns to process
// =============================================================

void ggml_gemm_i2_i8_s_interleaved(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx_base, int64_t row_start, int64_t a_row_bytes,
    const void * GGML_RESTRICT vy, int nr, int nc) {
#if defined(__AVX2__)
    // For the interleaved layout, the data for row `r` at byte offset `off` is at:
    //   group = r / 8
    //   group_base = group * 8 * a_row_bytes
    //   within_group = (r % 8) * INTERLEAVE + (off / INTERLEAVE) * 8 * INTERLEAVE + (off % INTERLEAVE)
    // With INTERLEAVE=8:
    //   within_group = (r % 8) * 8 + (off / 8) * 64 + (off % 8)
    //
    // But actually, since we process 4 rows that are consecutive (within a group of 8),
    // we can load them efficiently. For 4 consecutive rows r, r+1, r+2, r+3 within the
    // same group, at block `bl`:
    //   row_r:   group_base + bl * 64 + (r%8) * 8
    //   row_r+1: group_base + bl * 64 + ((r+1)%8) * 8
    //   etc.
    // These are 4 * 8 = 32 contiguous bytes if r%8 is 0 or 4!
    // For r%8 = 0: offsets 0,8,16,24 → we can load 32 bytes and deinterleave
    // For r%8 = 4: offsets 32,40,48,56 → same, load 32 bytes

    const uint8_t * x = (const uint8_t *)vx_base;
    const int8_t  * y = (const int8_t *)vy;
    const int64_t nb_128 = n / 128; // number of 128-element blocks
    const int64_t INTERLEAVE = 8;
    const int64_t NROWS_GROUP = 8;
    const int64_t group_stride = NROWS_GROUP * a_row_bytes; // bytes per group of 8 rows

    const __m256i mask_2bit = _mm256_set1_epi8(0x03);
    const __m256i one16 = _mm256_set1_epi16(1);

    int64_t r = 0;
    for (; r + 4 <= nc; r += 4) {
        const int64_t abs_row = row_start + r;
        const int64_t group = abs_row / NROWS_GROUP;
        const int64_t row_in_group = abs_row % NROWS_GROUP;
        const uint8_t * group_base = x + group * group_stride;

        int64_t c = 0;
        for (; c + 4 <= nr; c += 4) {
            __m256i acc[4][4];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    acc[i][j] = _mm256_setzero_si256();

            for (int64_t bl = 0; bl < nb_128; bl++) {
                // Each 128-element block spans 32 packed bytes per row
                // In interleaved layout: 32 bytes = 4 sub-blocks of 8 bytes
                // Load 4 rows × 4 sub-blocks
                __m256i a_unpacked[4][4];
                for (int i = 0; i < 4; i++) {
                    // Load 32 bytes for row (row_in_group+i) from interleaved data
                    // These are at: group_base + (sub_block * 64 + (row_in_group+i) * 8) for each sub_block
                    const int64_t row_off = row_in_group + i;
                    uint8_t tmp_packed[32];
                    for (int sb = 0; sb < 4; sb++) {
                        const uint8_t * src = group_base + (bl * 4 + sb) * NROWS_GROUP * INTERLEAVE + row_off * INTERLEAVE;
                        memcpy(tmp_packed + sb * 8, src, 8);
                    }
                    __m256i packed = _mm256_loadu_si256((const __m256i *)tmp_packed);
                    a_unpacked[i][0] = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask_2bit);
                    a_unpacked[i][1] = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_2bit);
                    a_unpacked[i][2] = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask_2bit);
                    a_unpacked[i][3] = _mm256_and_si256(packed, mask_2bit);
                }

                for (int j = 0; j < 4; j++) {
                    const int8_t * b_ptr = y + (c + j) * n + bl * 128;
                    __m256i b0 = _mm256_loadu_si256((const __m256i *)(b_ptr));
                    __m256i b1 = _mm256_loadu_si256((const __m256i *)(b_ptr + 32));
                    __m256i b2 = _mm256_loadu_si256((const __m256i *)(b_ptr + 64));
                    __m256i b3 = _mm256_loadu_si256((const __m256i *)(b_ptr + 96));

                    for (int i = 0; i < 4; i++) {
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][0], b0);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][1], b1);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][2], b2);
                        acc[i][j] = _mm256_dpbusd_epi32(acc[i][j], a_unpacked[i][3], b3);
#else
                        __m256i d0 = _mm256_maddubs_epi16(a_unpacked[i][0], b0);
                        __m256i d1 = _mm256_maddubs_epi16(a_unpacked[i][1], b1);
                        __m256i d2 = _mm256_maddubs_epi16(a_unpacked[i][2], b2);
                        __m256i d3 = _mm256_maddubs_epi16(a_unpacked[i][3], b3);
                        __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(d0, d1),
                                                         _mm256_add_epi16(d2, d3));
                        acc[i][j] = _mm256_add_epi32(acc[i][j], _mm256_madd_epi16(sum16, one16));
#endif
                    }
                }
            }

            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    __m128i sum128 = _mm_add_epi32(
                        _mm256_castsi256_si128(acc[i][j]),
                        _mm256_extractf128_si256(acc[i][j], 1));
                    __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
                    __m128i sum64 = _mm_add_epi32(hi64, sum128);
                    __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
                    s[(c + j) * bs + (r + i)] = (float)_mm_cvtsi128_si32(
                        _mm_add_epi32(sum64, hi32));
                }
            }
        }

        // Remaining B columns
        for (; c < nr; c++) {
            __m256i acc_rem[4];
            for (int i = 0; i < 4; i++)
                acc_rem[i] = _mm256_setzero_si256();

            for (int64_t bl = 0; bl < nb_128; bl++) {
                const int8_t * b_ptr = y + c * n + bl * 128;
                __m256i b0 = _mm256_loadu_si256((const __m256i *)(b_ptr));
                __m256i b1 = _mm256_loadu_si256((const __m256i *)(b_ptr + 32));
                __m256i b2 = _mm256_loadu_si256((const __m256i *)(b_ptr + 64));
                __m256i b3 = _mm256_loadu_si256((const __m256i *)(b_ptr + 96));

                for (int i = 0; i < 4; i++) {
                    const int64_t row_off = row_in_group + i;
                    uint8_t tmp_packed[32];
                    for (int sb = 0; sb < 4; sb++) {
                        const uint8_t * src = group_base + (bl * 4 + sb) * NROWS_GROUP * INTERLEAVE + row_off * INTERLEAVE;
                        memcpy(tmp_packed + sb * 8, src, 8);
                    }
                    __m256i packed = _mm256_loadu_si256((const __m256i *)tmp_packed);
                    __m256i a0 = _mm256_and_si256(_mm256_srli_epi16(packed, 6), mask_2bit);
                    __m256i a1 = _mm256_and_si256(_mm256_srli_epi16(packed, 4), mask_2bit);
                    __m256i a2 = _mm256_and_si256(_mm256_srli_epi16(packed, 2), mask_2bit);
                    __m256i a3 = _mm256_and_si256(packed, mask_2bit);
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a0, b0);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a1, b1);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a2, b2);
                    acc_rem[i] = _mm256_dpbusd_epi32(acc_rem[i], a3, b3);
#else
                    __m256i d0 = _mm256_maddubs_epi16(a0, b0);
                    __m256i d1 = _mm256_maddubs_epi16(a1, b1);
                    __m256i d2 = _mm256_maddubs_epi16(a2, b2);
                    __m256i d3 = _mm256_maddubs_epi16(a3, b3);
                    __m256i sum16 = _mm256_add_epi16(_mm256_add_epi16(d0, d1),
                                                     _mm256_add_epi16(d2, d3));
                    acc_rem[i] = _mm256_add_epi32(acc_rem[i], _mm256_madd_epi16(sum16, one16));
#endif
                }
            }

            for (int i = 0; i < 4; i++) {
                __m128i sum128 = _mm_add_epi32(
                    _mm256_castsi256_si128(acc_rem[i]),
                    _mm256_extractf128_si256(acc_rem[i], 1));
                __m128i hi64 = _mm_unpackhi_epi64(sum128, sum128);
                __m128i sum64 = _mm_add_epi32(hi64, sum128);
                __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
                s[c * bs + (r + i)] = (float)_mm_cvtsi128_si32(
                    _mm_add_epi32(sum64, hi32));
            }
        }
    }

    // Remaining A rows (< 4) — fallback to non-interleaved single-row dot
    for (; r < nc; r++) {
        const int64_t abs_row = row_start + r;
        const int64_t group = abs_row / NROWS_GROUP;
        const int64_t row_in_group = abs_row % NROWS_GROUP;
        const uint8_t * group_base = x + group * group_stride;

        for (int64_t c2 = 0; c2 < nr; c2++) {
            int32_t sum = 0;
            for (int64_t bl = 0; bl < nb_128; bl++) {
                for (int sb = 0; sb < 4; sb++) {
                    const uint8_t * a_ptr = group_base + (bl * 4 + sb) * NROWS_GROUP * INTERLEAVE + row_in_group * INTERLEAVE;
                    const int8_t * b_ptr = y + c2 * n + bl * 128 + sb * 32;
                    for (int i = 0; i < 8; i++) {
                        uint8_t byte = a_ptr[i];
                        sum += ((byte >> 6) & 3) * b_ptr[i*4+0];
                        sum += ((byte >> 4) & 3) * b_ptr[i*4+1];
                        sum += ((byte >> 2) & 3) * b_ptr[i*4+2];
                        sum += ((byte >> 0) & 3) * b_ptr[i*4+3];
                    }
                }
            }
            s[c2 * bs + r] = (float)sum;
        }
    }
#else
    // Scalar fallback — not implemented for interleaved layout
    GGML_ASSERT(false && "interleaved I2_S GEMM requires AVX2");
    GGML_UNUSED(n); GGML_UNUSED(s); GGML_UNUSED(bs);
    GGML_UNUSED(vx_base); GGML_UNUSED(row_start); GGML_UNUSED(a_row_bytes);
    GGML_UNUSED(vy); GGML_UNUSED(nr); GGML_UNUSED(nc);
#endif
}

void ggml_gemv_i2_i8_s_interleaved(int n, float * GGML_RESTRICT s, size_t bs,
    const void * GGML_RESTRICT vx_base, int64_t row_start, int64_t a_row_bytes,
    const void * GGML_RESTRICT vy, int nr, int nc) {
    // GEMV is just GEMM with nr=1
    ggml_gemm_i2_i8_s_interleaved(n, s, bs, vx_base, row_start, a_row_bytes, vy, nr, nc);
}
