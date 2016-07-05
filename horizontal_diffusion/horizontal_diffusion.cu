#include "horizontal_diffusion.h"
#include "../repository.hpp"
#include "../utils.hpp"
#include "horizontal_diffusion_reference.hpp"
#include "../functions.hpp"

#define BLOCK_X_SIZE 32
#define BLOCK_Y_SIZE 8

#define HALO_BLOCK_X_MINUS 1
#define HALO_BLOCK_X_PLUS 1

#define HALO_BLOCK_Y_MINUS 1
#define HALO_BLOCK_Y_PLUS 1

#define PADDED_BOUNDARY 1

#define CACHE_SIZE (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS) * (BLOCK_Y_SIZE + 2)
#define CACHE_SIZE_IN (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS + 4) * (BLOCK_Y_SIZE + 2 + 4)

inline __device__ unsigned int cache_index(const int ipos, const int jpos) {
    return (ipos + PADDED_BOUNDARY) +
           (jpos + HALO_BLOCK_Y_MINUS) * (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS);
}

__device__ unsigned int cache_index_in(const int ipos, const int jpos) {
    return (ipos + PADDED_BOUNDARY + 1) +
           (jpos + HALO_BLOCK_Y_MINUS + 1) * (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS + 2);
}

__global__ void cukernel(
    Real *in, Real *out, Real *coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides) {

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
    iblock_pos = -HALO_BLOCK_X_MINUS - 6;
    jblock_pos = -HALO_BLOCK_Y_MINUS - 6;
    if (threadIdx.y < jboundary_limit) {
        ipos = blockIdx.x * BLOCK_X_SIZE + threadIdx.x + halo.m_i;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.y - HALO_BLOCK_Y_MINUS + halo.m_j;
        iblock_pos = threadIdx.x;
        jblock_pos = threadIdx.y - HALO_BLOCK_Y_MINUS;
    } else if (threadIdx.y < iminus_limit &&
               threadIdx.x < (BLOCK_Y_SIZE + HALO_BLOCK_Y_MINUS + HALO_BLOCK_Y_PLUS) * PADDED_BOUNDARY) {
        ipos = blockIdx.x * BLOCK_X_SIZE - PADDED_BOUNDARY + threadIdx.x % PADDED_BOUNDARY + halo.m_i;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.x / PADDED_BOUNDARY + halo.m_j - HALO_BLOCK_Y_MINUS;
        iblock_pos = -PADDED_BOUNDARY + (int)threadIdx.x % PADDED_BOUNDARY;
        jblock_pos = threadIdx.x / PADDED_BOUNDARY - HALO_BLOCK_Y_MINUS;
    } else if (threadIdx.y < iplus_limit &&
               threadIdx.x < (BLOCK_Y_SIZE + HALO_BLOCK_Y_MINUS + HALO_BLOCK_Y_PLUS) * PADDED_BOUNDARY) {
        ipos = blockIdx.x * BLOCK_X_SIZE + threadIdx.x % PADDED_BOUNDARY + BLOCK_X_SIZE + halo.m_i;
        jpos = blockIdx.y * BLOCK_Y_SIZE + threadIdx.x / PADDED_BOUNDARY + halo.m_j - HALO_BLOCK_Y_MINUS;
        iblock_pos = threadIdx.x % PADDED_BOUNDARY + BLOCK_X_SIZE;
        jblock_pos = threadIdx.x / PADDED_BOUNDARY - HALO_BLOCK_Y_MINUS;
    }

    int index_ = index(ipos, jpos, 0, strides);

    // flx and fly can be defined with smaller cache sizes, however in order to reuse the same cache_index function, I
    // defined them here
    // with same size. shared memory pressure should not be too high nevertheless
    __shared__ Real lap[CACHE_SIZE];
    __shared__ Real flx[CACHE_SIZE];
    __shared__ Real fly[CACHE_SIZE];

    __shared__ Real in_cache[CACHE_SIZE_IN];

    Real in_ref[2];

    if (is_in_domain< -2, 2, -2, 2 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
        in_cache[cache_index_in(iblock_pos, jblock_pos)] = __ldg(&in[index_]);
        if (threadIdx.y == 0) {
            in_cache[cache_index_in(iblock_pos, jblock_pos - 1)] = __ldg(&in[index_ - index(0, 1, 0, strides)]);
        } else if (threadIdx.y == jboundary_limit - 1) {
            in_cache[cache_index_in(iblock_pos, jblock_pos + 1)] = __ldg(&in[index_ + index(0, 1, 0, strides)]);
        } else if (threadIdx.y == jboundary_limit) {
            in_cache[cache_index_in(iblock_pos - 1, jblock_pos)] = __ldg(&in[index_ - index(1, 0, 0, strides)]);
        } else if (threadIdx.y == jboundary_limit + 1) {
            in_cache[cache_index_in(iblock_pos + 1, jblock_pos)] = __ldg(&in[index_ + index(1, 0, 0, strides)]);
        }
    }

    __syncthreads();

    for (int kpos = 0; kpos < domain.m_k; ++kpos) {

        if (is_in_domain< -2, 2, -2, 2 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            in_ref[0] = __ldg(&in[index_ + index(0, 0, 1, strides)]);
            if (threadIdx.y == 0) {
                in_ref[1] = __ldg(&in[index_ - index(0, 1, 0, strides) + index(0, 0, 1, strides)]);
            } else if (threadIdx.y == jboundary_limit - 1) {
                in_ref[1] = __ldg(&in[index_ + index(0, 1, 1, strides)]);
            } else if (threadIdx.y == jboundary_limit) {
                in_ref[1] = __ldg(&in[index_ - index(1, 0, 0, strides) + index(0, 0, 1, strides)]);
            } else if (threadIdx.y == jboundary_limit + 1) {
                in_ref[1] = __ldg(&in[index_ + index(1, 0, 1, strides)]);
            }
        }

        //__syncthreads();

        if (is_in_domain< -1, 1, -1, 1 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {

            lap[cache_index(iblock_pos, jblock_pos)] = (Real)4 * in_cache[cache_index_in(iblock_pos, jblock_pos)] -
                                                       (in_cache[cache_index_in(iblock_pos + 1, jblock_pos)] +
                                                           in_cache[cache_index_in(iblock_pos - 1, jblock_pos)] +
                                                           in_cache[cache_index_in(iblock_pos, jblock_pos + 1)] +
                                                           in_cache[cache_index_in(iblock_pos, jblock_pos - 1)]);
        }

        __syncthreads();

        if (is_in_domain< -1, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            flx[cache_index(iblock_pos, jblock_pos)] =
                lap[cache_index(iblock_pos + 1, jblock_pos)] - lap[cache_index(iblock_pos, jblock_pos)];
            if (flx[cache_index(iblock_pos, jblock_pos)] * (in_cache[cache_index_in(iblock_pos + 1, jblock_pos)] -
                                                               in_cache[cache_index_in(iblock_pos, jblock_pos)]) >
                0) {
                flx[cache_index(iblock_pos, jblock_pos)] = 0.;
            }
        }

        if (is_in_domain< 0, 0, -1, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            fly[cache_index(iblock_pos, jblock_pos)] =
                lap[cache_index(iblock_pos, jblock_pos + 1)] - lap[cache_index(iblock_pos, jblock_pos)];
            if (fly[cache_index(iblock_pos, jblock_pos)] * (in_cache[cache_index_in(iblock_pos, jblock_pos + 1)] -
                                                               in_cache[cache_index_in(iblock_pos, jblock_pos)]) >
                0) {
                fly[cache_index(iblock_pos, jblock_pos)] = 0.;
            }
        }

        __syncthreads();

        if (is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            out[index_] =
                in_cache[cache_index_in(iblock_pos, jblock_pos)] -
                coeff[index_] *
                    (flx[cache_index(iblock_pos, jblock_pos)] - flx[cache_index(iblock_pos - 1, jblock_pos)] +
                        fly[cache_index(iblock_pos, jblock_pos)] - fly[cache_index(iblock_pos, jblock_pos - 1)]);
        }

        __syncthreads();

        if (is_in_domain< -2, 2, -2, 2 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {

            in_cache[cache_index_in(iblock_pos, jblock_pos)] = in_ref[0];
            if (threadIdx.y == 0) {
                in_cache[cache_index_in(iblock_pos, jblock_pos - 1)] = in_ref[1];
            } else if (threadIdx.y == jboundary_limit - 1) {
                in_cache[cache_index_in(iblock_pos, jblock_pos + 1)] = in_ref[1];
            } else if (threadIdx.y == jboundary_limit) {
                in_cache[cache_index_in(iblock_pos - 1, jblock_pos)] = in_ref[1];
            } else if (threadIdx.y == jboundary_limit + 1) {
                in_cache[cache_index_in(iblock_pos + 1, jblock_pos)] = in_ref[1];
            }
        }

        __syncthreads();

        index_ += index(0, 0, 1, strides);
    }
}

void launch_kernel(repository &repo, timer_cuda *time) {
    IJKSize domain = repo.domain();
    IJKSize halo = repo.halo();

    dim3 threads, blocks;
    threads.x = BLOCK_X_SIZE;
    threads.y = BLOCK_Y_SIZE + HALO_BLOCK_Y_MINUS + HALO_BLOCK_Y_PLUS + (HALO_BLOCK_X_MINUS > 0 ? 1 : 0) +
                (HALO_BLOCK_X_PLUS > 0 ? 1 : 0);
    threads.z = 1;
    blocks.x = (domain.m_i + BLOCK_X_SIZE - 1) / BLOCK_X_SIZE;
    blocks.y = (domain.m_j + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE;
    blocks.z = 1;

    IJKSize strides;
    compute_strides(domain, halo, strides);

    Real *in = repo.field_d("u_in");
    Real *out = repo.field_d("u_out");
    Real *coeff = repo.field_d("coeff");

    if (time)
        time->start();
    cukernel< < < blocks, threads, 0 > > >(in, out, coeff, domain, halo, strides);
    if (time)
        time->pause();
}
