#include "vertical_advection.h"
#include "../repository.hpp"
#include "../utils.hpp"
#include "vertical_advection_reference.hpp"
#include "../timer_cuda.hpp"

#define BLOCK_X_SIZE 32
#define BLOCK_Y_SIZE 8

#define HALO_BLOCK_X_MINUS 0
#define HALO_BLOCK_X_PLUS 0

#define HALO_BLOCK_Y_MINUS 0
#define HALO_BLOCK_Y_PLUS 0

#define PADDED_BOUNDARY 0

inline __device__ unsigned int cache_index(const unsigned int ipos, const unsigned int jpos) {
    return (ipos + PADDED_BOUNDARY) +
           (jpos + HALO_BLOCK_Y_MINUS) * (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS);
}

__global__ void cukernel(Real *u_stage,
    Real *wcon,
    Real *u_pos,
    Real *utens,
    Real *ccol,
    Real *dcol,
    Real *datacol,
    Real *utens_stage,
    Real *v_stage,
    Real *v_pos,
    Real *vtens,
    Real *vtens_stage,
    Real *w_stage,
    Real *w_pos,
    Real *wtens,
    Real *wtens_stage,
    const IJKSize domain,
    const IJKSize halo,
    const IJKSize strides) {

    const IJKSize strides_ = strides;
    unsigned int ipos, jpos;
    int iblock_pos, jblock_pos;
    const unsigned int jboundary_limit = BLOCK_Y_SIZE + HALO_BLOCK_Y_MINUS + HALO_BLOCK_Y_PLUS;
    const unsigned int iminus_limit = jboundary_limit + HALO_BLOCK_X_MINUS;
    const unsigned int iplus_limit = iminus_limit + HALO_BLOCK_X_PLUS;

    const unsigned int block_size_i =
        (blockIdx.x + 1) * BLOCK_X_SIZE < domain.m_i ? BLOCK_X_SIZE : domain.m_i - blockIdx.x * BLOCK_X_SIZE;
    const unsigned int block_size_j =
        (blockIdx.y + 1) * BLOCK_Y_SIZE < domain.m_j ? BLOCK_Y_SIZE : domain.m_j - blockIdx.y * BLOCK_Y_SIZE;

    // set the thread position by default out of the block
    iblock_pos = -HALO_BLOCK_X_MINUS - 1;
    jblock_pos = -HALO_BLOCK_Y_MINUS - 1;
    if (threadIdx.y < jboundary_limit) {
        ipos = blockIdx.x * BLOCK_X_SIZE + threadIdx.x + halo.m_i;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.y - HALO_BLOCK_Y_MINUS + halo.m_j;
        iblock_pos = threadIdx.x;
        jblock_pos = threadIdx.y - HALO_BLOCK_Y_MINUS;
    } else if (threadIdx.y < iminus_limit && threadIdx.x < BLOCK_Y_SIZE * PADDED_BOUNDARY) {
        ipos = blockIdx.x * BLOCK_X_SIZE - PADDED_BOUNDARY + threadIdx.x % PADDED_BOUNDARY + halo.m_i;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.x / PADDED_BOUNDARY + halo.m_j;
        iblock_pos = -PADDED_BOUNDARY + (int)threadIdx.x % PADDED_BOUNDARY;
        jblock_pos = threadIdx.x / PADDED_BOUNDARY;
    } else if (threadIdx.y < iplus_limit && threadIdx.x < BLOCK_Y_SIZE * PADDED_BOUNDARY) {
        ipos = blockIdx.x * BLOCK_X_SIZE + threadIdx.x % PADDED_BOUNDARY + BLOCK_X_SIZE + halo.m_i;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.x / PADDED_BOUNDARY + halo.m_j;
        iblock_pos = threadIdx.x % PADDED_BOUNDARY + BLOCK_X_SIZE;
        jblock_pos = threadIdx.x / PADDED_BOUNDARY;
    }
    forward_sweep(ipos,
        jpos,
        iblock_pos,
        jblock_pos,
        block_size_i,
        block_size_j,
        1,
        0,
        ccol,
        dcol,
        wcon,
        u_stage,
        u_pos,
        utens,
        utens_stage,
        domain,
        strides_);
    backward_sweep(ipos,
        jpos,
        iblock_pos,
        jblock_pos,
        block_size_i,
        block_size_j,
        ccol,
        dcol,
        datacol,
        u_pos,
        utens_stage,
        domain,
        strides_);

    forward_sweep(ipos,
        jpos,
        iblock_pos,
        jblock_pos,
        block_size_i,
        block_size_j,
        1,
        0,
        ccol,
        dcol,
        wcon,
        v_stage,
        v_pos,
        vtens,
        vtens_stage,
        domain,
        strides_);
    backward_sweep(ipos,
        jpos,
        iblock_pos,
        jblock_pos,
        block_size_i,
        block_size_j,
        ccol,
        dcol,
        datacol,
        v_pos,
        vtens_stage,
        domain,
        strides_);

    forward_sweep(ipos,
        jpos,
        iblock_pos,
        jblock_pos,
        block_size_i,
        block_size_j,
        1,
        0,
        ccol,
        dcol,
        wcon,
        w_stage,
        w_pos,
        wtens,
        wtens_stage,
        domain,
        strides_);
    backward_sweep(ipos,
        jpos,
        iblock_pos,
        jblock_pos,
        block_size_i,
        block_size_j,
        ccol,
        dcol,
        datacol,
        w_pos,
        wtens_stage,
        domain,
        strides_);
}

void launch_kernel(repository &repo, timer_cuda* time) {
    IJKSize domain = repo.domain();
    IJKSize halo = repo.halo();

    dim3 threads, blocks;
    threads.x = BLOCK_X_SIZE;
    threads.y = BLOCK_Y_SIZE;
    threads.z = 1;
    blocks.x = (domain.m_i + BLOCK_X_SIZE - 1) / BLOCK_X_SIZE;
    blocks.y = (domain.m_j + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE;
    blocks.z = 1;

    IJKSize strides;
    compute_strides(domain, halo, strides);

    Real *u_stage = repo.field_d("u_stage");
    Real *wcon = repo.field_d("wcon");
    Real *u_pos = repo.field_d("u_pos");
    Real *utens = repo.field_d("utens");
    Real *utens_stage = repo.field_d("utens_stage");

    Real *v_stage = repo.field_d("v_stage");
    Real *v_pos = repo.field_d("v_pos");
    Real *vtens = repo.field_d("vtens");
    Real *vtens_stage = repo.field_d("vtens_stage");

    Real *w_stage = repo.field_d("w_stage");
    Real *w_pos = repo.field_d("w_pos");
    Real *wtens = repo.field_d("wtens");
    Real *wtens_stage = repo.field_d("wtens_stage");

    Real *ccol = repo.field_d("ccol");
    Real *dcol = repo.field_d("dcol");
    Real *datacol = repo.field_d("datacol");

    if(time) time->start();
    cukernel<<< blocks, threads, 0 >>>(u_stage,
        wcon,
        u_pos,
        utens,
        ccol,
        dcol,
        datacol,
        utens_stage,
        v_stage,
        v_pos,
        vtens,
        vtens_stage,
        w_stage,
        w_pos,
        wtens,
        wtens_stage,
        domain,
        halo,
        strides);
    if(time) time->pause();
}
