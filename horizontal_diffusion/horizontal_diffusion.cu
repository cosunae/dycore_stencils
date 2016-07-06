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

inline __device__ unsigned int cache_index(const unsigned int ipos, const unsigned int jpos) {
    return (ipos + PADDED_BOUNDARY) +
           (jpos + HALO_BLOCK_Y_MINUS) * (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS);
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

    int index_ = index(ipos, jpos, 0, strides);

// flx and fly can be defined with smaller cache sizes, however in order to reuse the same cache_index function, I
// defined them here
// with same size. shared memory pressure should not be too high nevertheless
#define CACHE_SIZE (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS) * (BLOCK_Y_SIZE + 2)
    __shared__ Real lap[CACHE_SIZE];
    __shared__ Real flx[CACHE_SIZE];
    __shared__ Real fly[CACHE_SIZE];

    for (int kpos = 0; kpos < domain.m_k; ++kpos) {
        Real in_r_00, in_r_m0, in_r_p0, in_r_0m, in_r_0p;
        if (is_in_domain< -1, 1, -1, 1 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            in_r_00 = __ldg(& in[index_] );
            in_r_p0 = __ldg(& in[index_+index(1, 0,0, strides)] );
            in_r_m0 = __ldg(& in[index_-index(1, 0,0, strides)] );
            in_r_0p = __ldg(& in[index_+index(0, 1,0, strides)] );
            in_r_0m = __ldg(& in[index_-index(0, 1,0, strides)] );
            lap[cache_index(iblock_pos, jblock_pos)] =
                (Real)4 * in_r_00 - ( in_r_p0 + in_r_m0 + in_r_0p + in_r_0m );
        }

        __syncthreads();

        if (is_in_domain< -1, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            flx[cache_index(iblock_pos, jblock_pos)] =
                lap[cache_index(iblock_pos + 1, jblock_pos)] - lap[cache_index(iblock_pos, jblock_pos)];
            if (flx[cache_index(iblock_pos, jblock_pos)] *
                    (in_r_p0 - in_r_00) >
                0) {
                flx[cache_index(iblock_pos, jblock_pos)] = 0.;
            }
        }

        if (is_in_domain< 0, 0, -1, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            fly[cache_index(iblock_pos, jblock_pos)] =
                lap[cache_index(iblock_pos, jblock_pos + 1)] - lap[cache_index(iblock_pos, jblock_pos)];
            if (fly[cache_index(iblock_pos, jblock_pos)] *
                    (in_r_0p - in_r_00) >
                0) {
                fly[cache_index(iblock_pos, jblock_pos)] = 0.;
            }
        }

        __syncthreads();

        if (is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            out[index_] =
                in_r_00 -
                __ldg(&coeff[index_]) *
                    (flx[cache_index(iblock_pos, jblock_pos)] - flx[cache_index(iblock_pos - 1, jblock_pos)] +
                        fly[cache_index(iblock_pos, jblock_pos)] - fly[cache_index(iblock_pos, jblock_pos - 1)]);
        }

        index_ += index(0,0,1, strides);
    }
}

void launch_kernel(repository &repo, timer_cuda* time) {
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

    if(time) time->start();
    cukernel<<< blocks, threads, 0 >>>(in, out, coeff, domain, halo, strides);
    if(time) time->pause();
}


template<int tx, int ty>
__global__ void cukernel2_base(
    const Real *__restrict__ in, Real *out, const Real *__restrict__ coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides) {
    const int baseX = (blockDim.x - 2*halo.m_i)*blockIdx.x;
    const int baseY = (blockDim.y - 2*halo.m_j)*blockIdx.y;
    const int localX = threadIdx.x;
    const int localY = threadIdx.y;
    const int globalX = baseX + localX;
    const int globalY = baseY + localY;

    const int dx = 1;
    const int dy = blockDim.x;
    const int gx = strides.m_i;
    const int gy = strides.m_j;

    const int localIdx = localX + localY*blockDim.x;
    const int globalIdx0 = globalX*strides.m_i + globalY*strides.m_j;

    if(globalX >= domain.m_i || globalY >= domain.m_j) {
        return ;
    }

    __shared__ Real lap[tx*ty];
    __shared__ Real flx[tx*ty];
    __shared__ Real fly[tx*ty];



    for (int kpos = 0; kpos < domain.m_k; ++kpos) {
        int globalIdx = globalIdx0 + kpos*strides.m_k;
        Real in_r_00, in_r_p0, in_r_m0, in_r_0p, in_r_0m;

        if(localX>=1 && localY>=1 && localX < tx-1 && localY < ty-1) {
            in_r_00 = in[globalIdx];
            in_r_p0 = in[globalIdx+gx];
            in_r_m0 = in[globalIdx-gx];
            in_r_0p = in[globalIdx+gy];
            in_r_0m = in[globalIdx-gy];

            lap[localIdx] = ((Real)4)*in_r_00 - (in_r_p0 + in_r_m0) - (in_r_0p + in_r_0m);
        }
        __syncthreads();
        if(localX>=1 && localY>=1 && localX < tx-2 && localY < ty-1) {
            flx[localIdx] = lap[localIdx+dx] - lap[localIdx];
            if(flx[localIdx] * (in_r_p0 - in_r_00) > 0.0) { flx[localIdx] = 0.0; }
        }

        if(localX>=1 && localY>=1 && localX < tx-1 && localY < ty-2) {
            fly[localIdx] = lap[localIdx+dy] - lap[localIdx];
            if(fly[localIdx] * (in_r_0p - in_r_00) > 0.0) { fly[localIdx] = 0.0; }
        }

        __syncthreads();
        if(localX>=2 && localY>=2 && localX < tx-2 && localY < ty-2) {
            out[globalIdx] = in_r_00 -
                coeff[globalIdx] *
                (flx[localIdx] - flx[localIdx-dx] +
                 fly[localIdx] - fly[localIdx-dy]);
        }

    }

}

template<int tx, int ty>
__global__ void cukernel2_preload(
    const Real *__restrict__ in, Real *out, const Real *__restrict__ coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides) {
    const int baseX = (blockDim.x - 2*halo.m_i)*blockIdx.x;
    const int baseY = (blockDim.y - 2*halo.m_j)*blockIdx.y;
    const int localX = threadIdx.x;
    const int localY = threadIdx.y;
    const int globalX = baseX + localX;
    const int globalY = baseY + localY;

    const int dx = 1;
    const int dy = blockDim.x;

    const int localIdx = localX + localY*blockDim.x;
    const int globalIdx0 = globalX*strides.m_i + globalY*strides.m_j;

    if(globalX >= domain.m_i || globalY >= domain.m_j) {
        return ;
    }


    __shared__ Real lap[tx*ty];
    __shared__ Real flx[tx*ty];
    __shared__ Real fly[tx*ty];
    Real *in_s = flx;


    for (int kpos = 0; kpos < domain.m_k; ++kpos) {
        int globalIdx = globalIdx0 + kpos*strides.m_k;
        Real in_r_00, in_r_p0, in_r_m0, in_r_0p, in_r_0m;

        in_s[localIdx] = in_r_00 = in[globalIdx];
        Real coeff_r = coeff[globalIdx];
        __syncthreads();

        if(localX>=1 && localY>=1 && localX < tx-1 && localY < ty-1) {
            in_r_p0 = in_s[localIdx+dx];
            in_r_m0 = in_s[localIdx-dx];
            in_r_0p = in_s[localIdx+dy];
            in_r_0m = in_s[localIdx-dy];

            lap[localIdx] = ((Real)4)*in_r_00 - (in_r_p0 + in_r_m0) - (in_r_0p + in_r_0m);
        }
        __syncthreads();
        if(localX>=1 && localY>=1 && localX < tx-2 && localY < ty-1) {
            flx[localIdx] = lap[localIdx+dx] - lap[localIdx];
            if(flx[localIdx] * (in_r_p0 - in_r_00) > 0.0) { flx[localIdx] = 0.0; }
        }

        if(localX>=1 && localY>=1 && localX < tx-1 && localY < ty-2) {
            fly[localIdx] = lap[localIdx+dy] - lap[localIdx];
            if(fly[localIdx] * (in_r_0p - in_r_00) > 0.0) { fly[localIdx] = 0.0; }
        }

        __syncthreads();
        if(localX>=2 && localY>=2 && localX < tx-2 && localY < ty-2) {
            out[globalIdx] = in_r_00 -
                coeff_r *
                (flx[localIdx] - flx[localIdx-dx] +
                 fly[localIdx] - fly[localIdx-dy]);
        }

    }

}

template<int tx, int ty>
__global__ void cukernel2(
    const Real *__restrict__ in, Real *out, const Real *__restrict__ coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides) {
    const int baseX = (blockDim.x - 2*halo.m_i)*blockIdx.x;
    const int baseY = (blockDim.y - 2*halo.m_j)*blockIdx.y;
    const int localX = threadIdx.x;
    const int localY = threadIdx.y;
    const int globalX = baseX + localX;
    const int globalY = baseY + localY;

    const int dx = 1;
    const int dy = blockDim.x;

    const int localIdx = localX + localY*blockDim.x;
    const int globalIdx0 = globalX*strides.m_i + globalY*strides.m_j;

    if(globalX >= domain.m_i || globalY >= domain.m_j) {
        return ;
    }

    __shared__ Real in_s[tx*ty];
    __shared__ Real lap[tx*ty];
    __shared__ Real flx[tx*ty];
    __shared__ Real fly[tx*ty];

    Real in_r, lap_r, flx_r, fly_r, coeff_r;

    in_s[localIdx] = in_r = in[globalIdx0];
    coeff_r = coeff[globalIdx0];

    for (int kpos = 0; kpos < domain.m_k; ++kpos) {
        int globalIdx = globalIdx0 + kpos*strides.m_k;

        Real in_next = 0, coeff_next = 0;

        if(kpos<domain.m_k-1) {
            in_next = in[globalIdx + strides.m_k];
            coeff_next = coeff[globalIdx + strides.m_k];
        }
        __syncthreads();

        Real in_plus_dx, in_plus_dy;
        if(localX>=1 && localY>=1 && localX < tx-1 && localY < ty-1) {
            in_plus_dx = in_s[localIdx+dx];
            in_plus_dy = in_s[localIdx+dy];
            lap[localIdx] = lap_r = ((Real)4)*in_r - (in_plus_dx + in_s[localIdx-dx]) - (in_plus_dy + in_s[localIdx-dy]);
        }
        __syncthreads();
        if(localX>=1 && localY>=1 && localX < tx-2 && localY < ty-1) {
            flx_r = lap[localIdx+dx] - lap_r;
            if(flx_r * (in_plus_dx - in_r) > 0.0) { flx_r = 0.0; }
            flx[localIdx] = flx_r;
        }

        if(localX>=1 && localY>=1 && localX < tx-1 && localY < ty-2) {
            fly_r = lap[localIdx+dy] - lap_r;
            if(fly_r * (in_plus_dy - in_r) > 0.0) { fly_r = 0.0; }
            fly[localIdx] = fly_r;
        }

        __syncthreads();
        if(localX>=2 && localY>=2 && localX < tx-2 && localY < ty-2) {
            out[globalIdx] = in_r - coeff_r *
                ((flx_r - flx[localIdx-dx]) + (fly_r - fly[localIdx-dy]));
        }

        in_s[localIdx] = in_r = in_next;
        coeff_r = coeff_next;

    }

}



static const int tx = 36;
static const int ty = 14;

void (*kernels[])(const Real *__restrict__ in, Real *out, const Real *__restrict__ coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides) = {
    cukernel2_base<tx, ty>,
    cukernel2_preload<tx, ty>,
    cukernel2<tx, ty>,
};

int kernel_count = 3;

void launch_kernel2(repository &repo, timer_cuda* time, int version) {
    IJKSize domain = repo.domain();
    IJKSize halo = repo.halo();

    dim3 threads, blocks;
    threads.x = tx;
    threads.y = ty;
    threads.z = 1;
    blocks.x = (domain.m_i - 1) / (tx-4) + 1;
    blocks.y = (domain.m_j - 1) / (ty-4) + 1;
    blocks.z = 1;

    IJKSize strides;
    compute_strides(domain, halo, strides);

    Real *in = repo.field_d("u_in");
    Real *out = repo.field_d("u_out");
    Real *coeff = repo.field_d("coeff");

    //cudaFuncSetSharedMemConfig(kernels[version], cudaSharedMemBankSizeEightByte);
    if(time) time->start();
    kernels[version]<<< blocks, threads, 0 >>>(in, out, coeff, domain, halo, strides);
    if(time) time->pause();
}


template<unsigned int OX, unsigned int OY, unsigned int W> __device__ inline unsigned int X(unsigned int i) { return OX + i%W; }
template<unsigned int OX, unsigned int OY, unsigned int W> __device__ inline unsigned int Y(unsigned int i) { return OY + i/W; }
template<unsigned int OX, unsigned int OY, unsigned int W> __device__ inline unsigned int IDX(unsigned int x, unsigned int y) { return (x-OX) + (y-OY)*W; }


template<unsigned int THREADS, unsigned int BX, unsigned int BY>
__global__ void cukernel3_base(const Real *__restrict__ in, Real *out, const Real *__restrict__ coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides) {

    const unsigned int baseX = (BX - 2*halo.m_i) * blockIdx.x;
    const unsigned int baseY = (BY - 2*halo.m_j) * blockIdx.y;

    __shared__ Real in_shmem[BX*BY];
    __shared__ Real lat_shmem[BX*BY];
    __shared__ Real flx_shmem[BX*BY];
    __shared__ Real fly_shmem[BX*BY];

    for (int kpos = 0; kpos < domain.m_k; ++kpos) {
        const unsigned int globalOffset = strides.m_k*kpos;

        for(unsigned int i = threadIdx.x;i < BX*BY;i += blockDim.x) {
            const unsigned int globalX = baseX + X<0, 0, BX>(i);
            const unsigned int globalY = baseY + Y<0, 0, BX>(i);
            const unsigned int globalIdx = globalOffset + globalX*strides.m_i + globalY*strides.m_j;

            in_shmem[i] = __ldg(&in[globalIdx]);
        }

        __syncthreads();

        for(unsigned int i = threadIdx.x;i < (BX-2)*(BY-2);i += blockDim.x) {
            const unsigned int localX = X<1, 1, BX-2>(i);
            const unsigned int localY = Y<1, 1, BX-2>(i);

            lat_shmem[IDX<0,0,BX>(localX, localY)] = ((Real)4.0)*in_shmem[IDX<0,0,BX>(localX, localY)]
                - in_shmem[IDX<0,0,BX>(localX-1, localY)] - in_shmem[IDX<0,0,BX>(localX+1, localY)]
                - in_shmem[IDX<0,0,BX>(localX, localY-1)] - in_shmem[IDX<0,0,BX>(localX, localY+1)];
        }

        __syncthreads();

        for(unsigned int i = threadIdx.x;i < (BX-2)*(BY-2);i += blockDim.x) {
            const unsigned int localX = X<1, 1, BX-2>(i);
            const unsigned int localY = Y<1, 1, BX-2>(i);

            flx_shmem[IDX<0,0,BX>(localX, localY)] = lat_shmem[IDX<0,0,BX>(localX+1, localY)] - lat_shmem[IDX<0,0,BX>(localX, localY)];
            if(flx_shmem[IDX<0,0,BX>(localX, localY)] * (in_shmem[IDX<0,0,BX>(localX+1, localY)] - in_shmem[IDX<0,0,BX>(localX, localY)]) > 0.0) { flx_shmem[IDX<0,0,BX>(localX, localY)] = 0.0; }

            fly_shmem[IDX<0,0,BX>(localX, localY)] = lat_shmem[IDX<0,0,BX>(localX, localY+1)] - lat_shmem[IDX<0,0,BX>(localX, localY)];
            if(fly_shmem[IDX<0,0,BX>(localX, localY)] * (in_shmem[IDX<0,0,BX>(localX, localY+1)] - in_shmem[IDX<0,0,BX>(localX, localY)]) > 0.0) { fly_shmem[IDX<0,0,BX>(localX, localY)] = 0.0; }
        }

        __syncthreads();

        for(unsigned int i = threadIdx.x;i < (BX-4)*(BY-4);i += blockDim.x) {
            const unsigned int localX = X<2, 2, BX-4>(i);
            const unsigned int localY = Y<2, 2, BX-4>(i);
            const unsigned int globalX = baseX + localX;
            const unsigned int globalY = baseY + localY;
            const unsigned int globalIdx = globalOffset + globalX*strides.m_i + globalY*strides.m_j;

            out[globalIdx] = __ldg(&in[globalIdx]) - __ldg(&coeff[globalIdx]) *
                ((flx_shmem[IDX<0,0,BX>(localX, localY)] - flx_shmem[IDX<0,0,BX>(localX-1, localY)]) +
                 (fly_shmem[IDX<0,0,BX>(localX, localY)] - fly_shmem[IDX<0,0,BX>(localX, localY-1)]));
        }
        __syncthreads();
    }
}



void launch_kernel3(repository &repo, timer_cuda* time) {
    IJKSize domain = repo.domain();
    IJKSize halo = repo.halo();

    const int chunkx = 32;
    const int chunky = 8;
    const int threads = 128;

    dim3 blocks;
    blocks.x = (domain.m_i - 1) / (chunkx-4) + 1;
    blocks.y = (domain.m_j - 1) / (chunky-4) + 1;
    blocks.z = 1;

    IJKSize strides;
    compute_strides(domain, halo, strides);

    Real *in = repo.field_d("u_in");
    Real *out = repo.field_d("u_out");
    Real *coeff = repo.field_d("coeff");

    cudaFuncSetSharedMemConfig(cukernel3_base<threads, chunkx, chunky>, cudaSharedMemBankSizeEightByte);
    if(time) time->start();
    cukernel3_base<threads, chunkx, chunky><<< blocks, threads, 0 >>>(in, out, coeff, domain, halo, strides);
    if(time) time->pause();
}
