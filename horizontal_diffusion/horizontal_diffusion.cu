#include "horizontal_diffusion.h"
#include "../repository.hpp"
#include "../utils.hpp"
#include "horizontal_diffusion_reference.hpp"
#include "../functions.hpp"

#define BLOCK_X_SIZE 28
#define BLOCK_Y_SIZE 6

#define HALO_BLOCK_X_MINUS 1
#define HALO_BLOCK_X_PLUS 1

#define HALO_BLOCK_Y_MINUS 1
#define HALO_BLOCK_Y_PLUS 1

#define PADDED_BOUNDARY 1

#define __ldg( a ) a
// #define REF &
#define REF

inline __device__ unsigned int cache_index(const unsigned int ipos, const unsigned int jpos) {
    return (ipos) +
        (jpos) * ( BLOCK_X_SIZE
                   + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS);
}

inline __device__ unsigned int cache_index_in(const unsigned int ipos, const unsigned int jpos) {
    return (ipos) +
        (jpos) * ( BLOCK_X_SIZE
                   + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS +2 );
}

__global__ void cukernel(
    Real *in, Real *out, Real *coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides) {

    int ipos, jpos;
    // int iblock_pos, jblock_pos;
    // const unsigned int jboundary_limit = BLOCK_Y_SIZE + HALO_BLOCK_Y_MINUS + HALO_BLOCK_Y_PLUS;
    // const unsigned int iminus_limit = jboundary_limit + HALO_BLOCK_X_MINUS;
    // const unsigned int iplus_limit = iminus_limit + HALO_BLOCK_X_PLUS;

    // const unsigned int block_size_i =
    //     (blockIdx.x + 1) * BLOCK_X_SIZE < domain.m_i ? BLOCK_X_SIZE : domain.m_i - blockIdx.x * BLOCK_X_SIZE;
    // const unsigned int block_size_j =
    //     (blockIdx.y + 1) * BLOCK_Y_SIZE < domain.m_j ? BLOCK_Y_SIZE : domain.m_j - blockIdx.y * BLOCK_Y_SIZE;

    // // set the thread position by default out of the block
    // iblock_pos = -HALO_BLOCK_X_MINUS - 1;
    // jblock_pos = -HALO_BLOCK_Y_MINUS - 1;
    // if (threadIdx.y < jboundary_limit) {
    //     ipos = blockIdx.x * BLOCK_X_SIZE + threadIdx.x + halo.m_i;
    //     jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.y - HALO_BLOCK_Y_MINUS + halo.m_j;
    //     iblock_pos = threadIdx.x;
    //     jblock_pos = threadIdx.y - HALO_BLOCK_Y_MINUS;
    // } else if (threadIdx.y < iminus_limit && threadIdx.x < BLOCK_Y_SIZE * PADDED_BOUNDARY) {
    //     ipos = blockIdx.x * BLOCK_X_SIZE - PADDED_BOUNDARY + threadIdx.x % PADDED_BOUNDARY + halo.m_i;
    //     jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.x / PADDED_BOUNDARY + halo.m_j;
    //     iblock_pos = -PADDED_BOUNDARY + (int)threadIdx.x % PADDED_BOUNDARY;
    //     jblock_pos = threadIdx.x / PADDED_BOUNDARY;
    // } else if (threadIdx.y < iplus_limit && threadIdx.x < BLOCK_Y_SIZE * PADDED_BOUNDARY) {
    //     ipos = blockIdx.x * BLOCK_X_SIZE + threadIdx.x % PADDED_BOUNDARY + BLOCK_X_SIZE + halo.m_i;
    //     jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.x / PADDED_BOUNDARY + halo.m_j;
    //     iblock_pos = threadIdx.x % PADDED_BOUNDARY + BLOCK_X_SIZE;
    //     jblock_pos = threadIdx.x / PADDED_BOUNDARY;
    // }

    // set the thread position by default out of the block
    ipos = blockIdx.x * 28 + threadIdx.x;
    jpos = blockIdx.y * 6 + threadIdx.y;

    // int ipos_2 = blockIdx.x * 28 + threadIdx.x;
    // int jpos_2 = blockIdx.y * 6 + threadIdx.y;
    int index_ = index(ipos, jpos, 0, strides);
// flx and fly can be defined with smaller cache sizes, however in order to reuse the same cache_index function, I
// defined them here
// with same size. shared memory pressure should not be too high nevertheless
#define CACHE_SIZE ( BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS) * (BLOCK_Y_SIZE + 2)
#define CACHE_SIZE_IN ( BLOCK_X_SIZE+2 + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS) * (BLOCK_Y_SIZE+2 + 2)
    __shared__ Real in_s[CACHE_SIZE_IN];
    __shared__ Real lap[CACHE_SIZE];
    __shared__ Real flx[CACHE_SIZE];
    __shared__ Real fly[CACHE_SIZE];

    Real coeff_rp1 = coeff[index_];

    for (int kpos = 0; kpos < domain.m_k; ++kpos) {
        Real coeff_r = coeff_rp1;
        coeff_rp1 = coeff[index_+index(0,0,1,strides)];

        // if (is_in_domain< -2, 2, -2, 2 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
        if( ipos < domain.m_i && jpos < domain.m_j )
            in_s[cache_index_in(threadIdx.x, threadIdx.y)] = __ldg(REF in[index_]);
        // }
        __syncthreads();

        Real in_reg_ = __ldg(REF in[index_ - index(1, 0, 1, strides)]);

        if (// is_in_domain< -1, 1, -1, 1 >(iblock_pos, jblock_pos, block_size_i, block_size_j
            ipos<domain.m_i && jpos<domain.m_j && threadIdx.x>0 && threadIdx.x<31 && threadIdx.y>0 && threadIdx.y<7) {

            lap[cache_index(threadIdx.x, threadIdx.y)] =
                (Real)4 * __ldg( REF in[index_] ) -
                ( in_s[cache_index_in(threadIdx.x+1, threadIdx.y)] + in_s[cache_index_in(threadIdx.x-1, threadIdx.y)] +
                    in_s[cache_index_in(threadIdx.x, threadIdx.y+1)] + in_s[cache_index_in(threadIdx.x, threadIdx.y-1)]);
        }

        __syncthreads();

        if (//is_in_domain< -1, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)
            ipos<domain.m_i && jpos<domain.m_j && threadIdx.x>0 && threadIdx.x<30 && threadIdx.y>0+1 && threadIdx.y<7-1 ) {
            flx[cache_index(threadIdx.x, threadIdx.y)] =
                lap[cache_index(threadIdx.x+1, threadIdx.y)] - lap[cache_index(threadIdx.x, threadIdx.y)];
            if (flx[cache_index(threadIdx.x, threadIdx.y)] *
                (in_s[cache_index_in(threadIdx.x+1, threadIdx.y)] - in_s[cache_index_in(threadIdx.x, threadIdx.y)]) >
                0) {
                flx[cache_index(threadIdx.x, threadIdx.y)] = 0.;
            }
        }

        if (//is_in_domain< 0, 0, -1, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)
            ipos<domain.m_i && jpos<domain.m_j && threadIdx.x>0*+1 && threadIdx.x<31-1 && threadIdx.y>0 && threadIdx.y<6) {
            fly[cache_index(threadIdx.x, threadIdx.y)] =
                lap[cache_index(threadIdx.x, threadIdx.y + 1)] - lap[cache_index(threadIdx.x, threadIdx.y)];
            if (fly[cache_index(threadIdx.x, threadIdx.y)] *
                (in_s[cache_index_in(threadIdx.x, threadIdx.y+1)] - in_s[cache_index_in(threadIdx.x, threadIdx.y)]) >
                0) {
                fly[cache_index(threadIdx.x, threadIdx.y)] = 0.;
            }
        }

        __syncthreads();

        if (// is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)
            ipos<domain.m_i && jpos<domain.m_j && threadIdx.x>1 && threadIdx.x<30 && threadIdx.y>1 && threadIdx.y<6 ) {
            printf("index: %d", index_);
            out[index_] =
                in_s[cache_index_in(threadIdx.x, threadIdx.y)] -
                /*coeff_r*/coeff[index_] *
                    (flx[cache_index(threadIdx.x, threadIdx.y)] - flx[cache_index(threadIdx.x-1, threadIdx.y)] +
                        fly[cache_index(threadIdx.x, threadIdx.y)] - fly[cache_index(threadIdx.x, threadIdx.y - 1)]);
        }
        // if( ipos < domain.m_i && jpos < domain.m_j )
        // {
        //     printf("ipos %d, jpos %d, index_ %d, cache_index %d \n", ipos, jpos, index_, cache_index_in(threadIdx.x, threadIdx.y));
        //     out[index_] = in_s[cache_index_in(iblock_pos, jblock_pos)];
        // }

        index_ += index(0,0,1, strides);
    }
}

void launch_kernel(repository &repo, timer_cuda* time) {
    IJKSize domain = repo.domain();
    IJKSize halo = repo.halo();

    dim3 threads, blocks;
    threads.x = 32// BLOCK_X_SIZE
        ;
    threads.y = 8;//BLOCK_Y_SIZE + HALO_BLOCK_Y_MINUS + HALO_BLOCK_Y_PLUS + (HALO_BLOCK_X_MINUS > 0 ? 1 : 0) +
    //(HALO_BLOCK_X_PLUS > 0 ? 1 : 0);
    threads.z = 1;
    blocks.x = (domain.m_i + BLOCK_X_SIZE - 1) /  BLOCK_X_SIZE;
    blocks.y = (domain.m_j + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE;
    blocks.z = 1;

    IJKSize strides;
    compute_strides(domain, halo, strides);

    Real *in = repo.field_d("u_in");
    Real *out = repo.field_d("u_out");
    Real *coeff = repo.field_d("coeff");

    if(time) time->start();
    cukernel<<< blocks, threads, 0 >>>(in, out, coeff, domain, halo, strides);
    if(time) time->pause();
}
