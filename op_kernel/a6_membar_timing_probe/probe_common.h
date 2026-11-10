#ifndef A6_MEMBAR_PROBE_COMMON_H
#define A6_MEMBAR_PROBE_COMMON_H
#include <stdint.h>

#if !defined(__NPU_ARCH__) || __NPU_ARCH__ != 9201
#error "Compile this probe for A6 dav-920r1-vec"
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// UB: input [0,256), output [0x1000,0x1800), scratch [0x2000,0x2800).
// The scalar scatter descriptor occupies [0x3000,0x3020).
__attribute__((always_inline)) inline [aicore]
void probe_copy_in(__gm__ float *input) {
    __ubuf__ float *ub = (__ubuf__ float *)get_imm(0);
    nddma_desc desc(nddma_desc::loop_desc(64, 1, 1));
    nddma_out_to_ub(ub, input, 0, desc, 0, NEAREST_PADDING, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
}

__attribute__((always_inline)) inline [aicore]
void probe_copy_out(__gm__ float *output) {
    __ubuf__ uint32_t *ub = (__ubuf__ uint32_t *)get_imm(0x1000);
    __ubuf__ uint32_t *offsets = (__ubuf__ uint32_t *)get_imm(0x3000);
    offsets[0] = 0;
    pipe_barrier(PIPE_ALL);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    set_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID1);
    scatter_ubuf_to_gm((__gm__ uint32_t *)output, ub, 0, 0, offsets, 0, 2048, 0, 1);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
}
#endif
