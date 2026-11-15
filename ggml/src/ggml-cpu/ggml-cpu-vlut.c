// Vec-LUT ternary GEMM (I2_V_4 weights x per-token int8 activations), ported from
// vlut.cpp (https://github.com/Cipherxzc/vlut.cpp, MIT), arXiv:2512.06443.
// Fix vs upstream: the activation scale uses absmax, not the signed max.
#include "ggml-cpu-vlut.h"

#include "ggml-impl.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef GGML_VLUT_TABLE_ENTRY_SIZE
#define GGML_VLUT_TABLE_ENTRY_SIZE 8
#endif
#define TABLE_ENTRY_SIZE GGML_VLUT_TABLE_ENTRY_SIZE

#define UNUSED GGML_UNUSED
#define VLUT_EPS 1e-5

// Accumulate the transposed table entries
#define ACCUMULATE_TABLE_TRANS(ss, ss2, nc, entry_tile_remain, entry_tile_count) \
    do { \
        for (int j = 0; j < entry_tile_remain; j++) { \
            for (int i = 0; i < nc; i++) { \
                ss2[(entry_tile_count * TABLE_ENTRY_SIZE + j) * nc + i] += ss[i * TABLE_ENTRY_SIZE + j]; \
            } \
        } \
    } while(0)


#define MAKE_TABLES_I2S_BLOCK4(g)                                                                                    \
    do {                                                                                                             \
        gemm_make_table_i2v(table0, y_block + (((g)*group_size + (i * tile_size + 0) * 4) * TABLE_ENTRY_SIZE)); \
        gemm_make_table_i2v(table1, y_block + (((g)*group_size + (i * tile_size + 1) * 4) * TABLE_ENTRY_SIZE)); \
        gemm_make_table_i2v(table2, y_block + (((g)*group_size + (i * tile_size + 2) * 4) * TABLE_ENTRY_SIZE)); \
        gemm_make_table_i2v(table3, y_block + (((g)*group_size + (i * tile_size + 3) * 4) * TABLE_ENTRY_SIZE)); \
    } while (0)

#define ADD_TABLE_ENTRIES_BLOCK4(nc)                                           \
    do {                                                                       \
        for (int c = 0; c < (nc); c++) {                                       \
            uint8_t v0 = x_block[c * tile_size + 0];                            \
            uint8_t v1 = x_block[c * tile_size + 1];                            \
            uint8_t v2 = x_block[c * tile_size + 2];                            \
            uint8_t v3 = x_block[c * tile_size + 3];                            \
                                                                               \
            const int16_t *restrict rt0 = table0 + v0 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt1 = table1 + v1 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt2 = table2 + v2 * TABLE_ENTRY_SIZE; \
            const int16_t *restrict rt3 = table3 + v3 * TABLE_ENTRY_SIZE; \
                                                                               \
            int16_t *restrict rs = sum_i16 + c * TABLE_ENTRY_SIZE;             \
                                                                               \
            for (int r = 0; r < TABLE_ENTRY_SIZE; r++) {                       \
                rs[r] += rt0[r] + rt1[r] + rt2[r] + rt3[r];                    \
            }                                                                  \
        }                                                                      \
    } while (0)

inline static void add(int16_t *restrict t1, const int16_t *restrict t2, const int8_t *restrict y) {
    for (int i = 0; i < TABLE_ENTRY_SIZE; i++) {
        t1[i] = t2[i] + y[i];
    }
}

inline static void sub(int16_t *restrict t1, const int16_t *restrict t2, const int8_t *restrict y) {
    for (int i = 0; i < TABLE_ENTRY_SIZE; i++) {
        t1[i] = t2[i] - y[i];
    }
}

inline static void rev(int16_t *restrict t1, const int16_t *restrict t2) {
    for (int i = 0; i < TABLE_ENTRY_SIZE; i++) {
        t1[i] = -t2[i];
    }
}

static void gemm_make_table_i2v(int16_t *restrict table, const int8_t *restrict y) {
    const int8_t *restrict y0 = y;
    const int8_t *restrict y1 = y0 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y2 = y1 + TABLE_ENTRY_SIZE;
    const int8_t *restrict y3 = y2 + TABLE_ENTRY_SIZE;

    add(table + 41 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y0);
    add(table + 43 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y1);
    add(table + 49 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y2);
    add(table + 67 * TABLE_ENTRY_SIZE, table + 40 * TABLE_ENTRY_SIZE, y3);
    rev(table + 39 * TABLE_ENTRY_SIZE, table + 41 * TABLE_ENTRY_SIZE);
    rev(table + 37 * TABLE_ENTRY_SIZE, table + 43 * TABLE_ENTRY_SIZE);
    add(table + 44 * TABLE_ENTRY_SIZE, table + 43 * TABLE_ENTRY_SIZE, y0);
    rev(table + 31 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE);
    add(table + 50 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE, y0);
    add(table + 52 * TABLE_ENTRY_SIZE, table + 49 * TABLE_ENTRY_SIZE, y1);
    rev(table + 13 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE);
    add(table + 68 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y0);
    add(table + 70 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y1);
    add(table + 76 * TABLE_ENTRY_SIZE, table + 67 * TABLE_ENTRY_SIZE, y2);
    add(table + 38 * TABLE_ENTRY_SIZE, table + 37 * TABLE_ENTRY_SIZE, y0);
    rev(table + 36 * TABLE_ENTRY_SIZE, table + 44 * TABLE_ENTRY_SIZE);
    add(table + 32 * TABLE_ENTRY_SIZE, table + 31 * TABLE_ENTRY_SIZE, y0);
    add(table + 34 * TABLE_ENTRY_SIZE, table + 31 * TABLE_ENTRY_SIZE, y1);
    rev(table + 30 * TABLE_ENTRY_SIZE, table + 50 * TABLE_ENTRY_SIZE);
    rev(table + 28 * TABLE_ENTRY_SIZE, table + 52 * TABLE_ENTRY_SIZE);
    add(table + 53 * TABLE_ENTRY_SIZE, table + 52 * TABLE_ENTRY_SIZE, y0);
    add(table + 14 * TABLE_ENTRY_SIZE, table + 13 * TABLE_ENTRY_SIZE, y0);
    add(table + 16 * TABLE_ENTRY_SIZE, table + 13 * TABLE_ENTRY_SIZE, y1);
    add(table + 22 * TABLE_ENTRY_SIZE, table + 13 * TABLE_ENTRY_SIZE, y2);
    rev(table + 12 * TABLE_ENTRY_SIZE, table + 68 * TABLE_ENTRY_SIZE);
    rev(table + 10 * TABLE_ENTRY_SIZE, table + 70 * TABLE_ENTRY_SIZE);
    add(table + 71 * TABLE_ENTRY_SIZE, table + 70 * TABLE_ENTRY_SIZE, y0);
    rev(table + 4 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE);
    add(table + 77 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y0);
    add(table + 79 * TABLE_ENTRY_SIZE, table + 76 * TABLE_ENTRY_SIZE, y1);
    rev(table + 42 * TABLE_ENTRY_SIZE, table + 38 * TABLE_ENTRY_SIZE);
    rev(table + 48 * TABLE_ENTRY_SIZE, table + 32 * TABLE_ENTRY_SIZE);
    add(table + 35 * TABLE_ENTRY_SIZE, table + 34 * TABLE_ENTRY_SIZE, y0);
    rev(table + 46 * TABLE_ENTRY_SIZE, table + 34 * TABLE_ENTRY_SIZE);
    add(table + 29 * TABLE_ENTRY_SIZE, table + 28 * TABLE_ENTRY_SIZE, y0);
    rev(table + 27 * TABLE_ENTRY_SIZE, table + 53 * TABLE_ENTRY_SIZE);
    rev(table + 66 * TABLE_ENTRY_SIZE, table + 14 * TABLE_ENTRY_SIZE);
    add(table + 17 * TABLE_ENTRY_SIZE, table + 16 * TABLE_ENTRY_SIZE, y0);
    rev(table + 64 * TABLE_ENTRY_SIZE, table + 16 * TABLE_ENTRY_SIZE);
    add(table + 23 * TABLE_ENTRY_SIZE, table + 22 * TABLE_ENTRY_SIZE, y0);
    add(table + 25 * TABLE_ENTRY_SIZE, table + 22 * TABLE_ENTRY_SIZE, y1);
    rev(table + 58 * TABLE_ENTRY_SIZE, table + 22 * TABLE_ENTRY_SIZE);
    add(table + 11 * TABLE_ENTRY_SIZE, table + 10 * TABLE_ENTRY_SIZE, y0);
    rev(table + 9 * TABLE_ENTRY_SIZE, table + 71 * TABLE_ENTRY_SIZE);
    add(table + 5 * TABLE_ENTRY_SIZE, table + 4 * TABLE_ENTRY_SIZE, y0);
    add(table + 7 * TABLE_ENTRY_SIZE, table + 4 * TABLE_ENTRY_SIZE, y1);
    rev(table + 3 * TABLE_ENTRY_SIZE, table + 77 * TABLE_ENTRY_SIZE);
    rev(table + 1 * TABLE_ENTRY_SIZE, table + 79 * TABLE_ENTRY_SIZE);
    add(table + 80 * TABLE_ENTRY_SIZE, table + 79 * TABLE_ENTRY_SIZE, y0);
    rev(table + 45 * TABLE_ENTRY_SIZE, table + 35 * TABLE_ENTRY_SIZE);
    add(table + 47 * TABLE_ENTRY_SIZE, table + 46 * TABLE_ENTRY_SIZE, y0);
    rev(table + 51 * TABLE_ENTRY_SIZE, table + 29 * TABLE_ENTRY_SIZE);
    rev(table + 63 * TABLE_ENTRY_SIZE, table + 17 * TABLE_ENTRY_SIZE);
    add(table + 65 * TABLE_ENTRY_SIZE, table + 64 * TABLE_ENTRY_SIZE, y0);
    rev(table + 57 * TABLE_ENTRY_SIZE, table + 23 * TABLE_ENTRY_SIZE);
    add(table + 26 * TABLE_ENTRY_SIZE, table + 25 * TABLE_ENTRY_SIZE, y0);
    rev(table + 55 * TABLE_ENTRY_SIZE, table + 25 * TABLE_ENTRY_SIZE);
    add(table + 59 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE, y0);
    add(table + 61 * TABLE_ENTRY_SIZE, table + 58 * TABLE_ENTRY_SIZE, y1);
    rev(table + 69 * TABLE_ENTRY_SIZE, table + 11 * TABLE_ENTRY_SIZE);
    rev(table + 75 * TABLE_ENTRY_SIZE, table + 5 * TABLE_ENTRY_SIZE);
    add(table + 8 * TABLE_ENTRY_SIZE, table + 7 * TABLE_ENTRY_SIZE, y0);
    rev(table + 73 * TABLE_ENTRY_SIZE, table + 7 * TABLE_ENTRY_SIZE);
    add(table + 2 * TABLE_ENTRY_SIZE, table + 1 * TABLE_ENTRY_SIZE, y0);
    rev(table + 0 * TABLE_ENTRY_SIZE, table + 80 * TABLE_ENTRY_SIZE);
    rev(table + 33 * TABLE_ENTRY_SIZE, table + 47 * TABLE_ENTRY_SIZE);
    rev(table + 15 * TABLE_ENTRY_SIZE, table + 65 * TABLE_ENTRY_SIZE);
    rev(table + 54 * TABLE_ENTRY_SIZE, table + 26 * TABLE_ENTRY_SIZE);
    add(table + 56 * TABLE_ENTRY_SIZE, table + 55 * TABLE_ENTRY_SIZE, y0);
    rev(table + 21 * TABLE_ENTRY_SIZE, table + 59 * TABLE_ENTRY_SIZE);
    rev(table + 19 * TABLE_ENTRY_SIZE, table + 61 * TABLE_ENTRY_SIZE);
    add(table + 62 * TABLE_ENTRY_SIZE, table + 61 * TABLE_ENTRY_SIZE, y0);
    rev(table + 72 * TABLE_ENTRY_SIZE, table + 8 * TABLE_ENTRY_SIZE);
    add(table + 74 * TABLE_ENTRY_SIZE, table + 73 * TABLE_ENTRY_SIZE, y0);
    rev(table + 78 * TABLE_ENTRY_SIZE, table + 2 * TABLE_ENTRY_SIZE);
    rev(table + 24 * TABLE_ENTRY_SIZE, table + 56 * TABLE_ENTRY_SIZE);
    add(table + 20 * TABLE_ENTRY_SIZE, table + 19 * TABLE_ENTRY_SIZE, y0);
    rev(table + 18 * TABLE_ENTRY_SIZE, table + 62 * TABLE_ENTRY_SIZE);
    rev(table + 6 * TABLE_ENTRY_SIZE, table + 74 * TABLE_ENTRY_SIZE);
    rev(table + 60 * TABLE_ENTRY_SIZE, table + 20 * TABLE_ENTRY_SIZE);
}


static void quantize_row_i8_v_tile(const float *x, void *y, int64_t n, float *scale) {
    int8_t *dst = (int8_t *)y;

    double max = VLUT_EPS;
    for (int i = 0; i < n; i++) {
        max = MAX(max, fabs((double)x[i])); /* absmax (was signed max) */
    }
    const double s = max / 127;
    const double is = 1e0 / MAX(s, VLUT_EPS);

    for (int i = 0, j = 0; i < n; i++, j += TABLE_ENTRY_SIZE) {
        float v = round((double)x[i] * is);
        if (v > 127) {
            v = 127;
        } else if (v < -128) {
            v = -128;
        }
        dst[j] = (int8_t)v;
    }

    *scale = (float)s;
}

static int64_t vlut_pad_rows(int64_t nr) {
    return (nr % TABLE_ENTRY_SIZE) ? nr + TABLE_ENTRY_SIZE - (nr % TABLE_ENTRY_SIZE) : nr;
}

size_t ggml_vlut_wsize(int64_t ne10, int64_t ne11) {
    return vlut_pad_rows(ne11) * ne10 * sizeof(int8_t) + ne11 * sizeof(float);
}

void ggml_vlut_quantize_act(const char * src, size_t nb11, int64_t ne10, int64_t ne11, void * wdata, int ith, int nth) {
    int8_t * w = (int8_t *) wdata;
    float * scale = (float *) (w + vlut_pad_rows(ne11) * ne10);
    for (int64_t i11 = ith; i11 < ne11; i11 += nth) {
        const int64_t i = i11 / TABLE_ENTRY_SIZE;
        const int64_t j = i11 % TABLE_ENTRY_SIZE;
        quantize_row_i8_v_tile((const float *) (src + i11 * nb11), w + i * TABLE_ENTRY_SIZE * ne10 + j, ne10, scale + i11);
    }
}

void ggml_gemm_i2v4_i8v_lut(int ith, int nth, int n, float *restrict s, size_t bs, const void *restrict vx,
                            const void *restrict vy, int nr, int nc) {
    UNUSED(ith);
    UNUSED(nth);

    // nr: src1->ne[1], nc: src0->ne[1]
    assert(n % 4 == 0);

    static const int tile_size = 4;

    int16_t *restrict sum_i16 = (int16_t *)malloc(sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    int32_t *restrict sum_i32 = (int32_t *)malloc(sizeof(int32_t) * nr * nc);
    int16_t *restrict table0 = (int16_t *)malloc((sizeof(int16_t) * TABLE_ENTRY_SIZE) * 81);
    int16_t *restrict table1 = (int16_t *)malloc((sizeof(int16_t) * TABLE_ENTRY_SIZE) * 81);
    int16_t *restrict table2 = (int16_t *)malloc((sizeof(int16_t) * TABLE_ENTRY_SIZE) * 81);
    int16_t *restrict table3 = (int16_t *)malloc((sizeof(int16_t) * TABLE_ENTRY_SIZE) * 81);

    memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
    memset(sum_i32, 0, sizeof(int32_t) * nr * nc);
    memset(table0 + 40 * TABLE_ENTRY_SIZE, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE);
    memset(table1 + 40 * TABLE_ENTRY_SIZE, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE);
    memset(table2 + 40 * TABLE_ENTRY_SIZE, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE);
    memset(table3 + 40 * TABLE_ENTRY_SIZE, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE);

    static const int group_size = 512;

    const int group_count = n / group_size;  // not including remains
    const int group_size_remain = n % group_size;
    const int entry_tile_count = nr / TABLE_ENTRY_SIZE;  // not including remains
    const int entry_tile_remain = nr % TABLE_ENTRY_SIZE;

    // tiles
    for (int t = 0; t < entry_tile_count; t++) {
        const int8_t *restrict y_block = (const int8_t *)vy + t * n * TABLE_ENTRY_SIZE;
        // groups
        for (int g = 0; g < group_count; g++) {
            int pack_cnt = group_size / 4 / tile_size;
            for (int i = 0; i < pack_cnt; i++) {
                MAKE_TABLES_I2S_BLOCK4(g);
                const uint8_t *restrict x_block = (const uint8_t *)vx + (g * group_size / 4 + i * tile_size) * bs;
                ADD_TABLE_ENTRIES_BLOCK4(nc);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
        // group remain
        if (group_size_remain > 0) {
            int pack_cnt = group_size_remain / 4 / tile_size;
            for (int i = 0; i < pack_cnt; i++) {
                MAKE_TABLES_I2S_BLOCK4(group_count);
                const uint8_t *restrict x_block = (const uint8_t *)vx + (group_count * group_size / 4 + i * tile_size) * bs;
                ADD_TABLE_ENTRIES_BLOCK4(nc);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, TABLE_ENTRY_SIZE, t);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }

    // tile remain
    if (entry_tile_remain > 0) {
        const int8_t *restrict y_block = (const int8_t *)vy + entry_tile_count * n * TABLE_ENTRY_SIZE;
        // groups
        for (int g = 0; g < group_count; g++) {
            int pack_cnt = group_size / 4 / tile_size;
            for (int i = 0; i < pack_cnt; i++) {
                MAKE_TABLES_I2S_BLOCK4(g);
                const uint8_t *restrict x_block = (const uint8_t *)vx + (g * group_size / 4 + i * tile_size) * bs;
                ADD_TABLE_ENTRIES_BLOCK4(nc);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
        // group remain
        if (group_size_remain > 0) {
            int pack_cnt = group_size_remain / 4 / tile_size;
            for (int i = 0; i < pack_cnt; i++) {
                MAKE_TABLES_I2S_BLOCK4(group_count);
                const uint8_t *restrict x_block = (const uint8_t *)vx + (group_count * group_size / 4 + i * tile_size) * bs;
                ADD_TABLE_ENTRIES_BLOCK4(nc);
            }
            ACCUMULATE_TABLE_TRANS(sum_i16, sum_i32, nc, entry_tile_remain, entry_tile_count);
            memset(sum_i16, 0, sizeof(int16_t) * TABLE_ENTRY_SIZE * nc);
        }
    }

    const size_t y_size = ((nr % TABLE_ENTRY_SIZE) ? nr + TABLE_ENTRY_SIZE - (nr % TABLE_ENTRY_SIZE) : nr) * n;
    const float *sc = (const float *)((const int8_t *)vy + y_size);
    for (int r = 0; r < nr; r++) {
        const float scale = sc[r];
        float *restrict sr = s + r * bs;
        const int32_t *restrict ss2r = sum_i32 + r * nc;
        for (int c = 0; c < nc; c++) {
            sr[c] = ss2r[c] * scale;
        }
    }

    free(table0);
    free(table1);
    free(table2);
    free(table3);
    free(sum_i16);
    free(sum_i32);
}
