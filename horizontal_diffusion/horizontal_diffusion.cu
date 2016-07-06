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

inline __device__ unsigned int cache_index_in(const unsigned int ipos, const unsigned int jpos) {
    // return (ipos + PADDED_BOUNDARY+1) +
    //        (jpos + HALO_BLOCK_Y_MINUS+1) * (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS+2);
    return (ipos) +
           (jpos) * (BLOCK_X_SIZE +4);
}

__global__ void cukernel(
    Real *in, Real *out, Real *coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides) {

    // unsigned int ipos, jpos;
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

    // int index_ = index(ipos, jpos, 0, strides);

    int ipos2 = blockIdx.x * BLOCK_X_SIZE + threadIdx.x;
    int jpos2 = blockIdx.y * BLOCK_Y_SIZE + threadIdx.y;
    // int index2_ = index(ipos2, jpos2, 0, strides);
    int index_base = index(blockIdx.x * BLOCK_X_SIZE, blockIdx.y * BLOCK_Y_SIZE, 0, strides);
    // int index_Y = index(blockIdx.x * BLOCK_X_SIZE, blockIdx.y * BLOCK_Y_SIZE + BLOCK_Y_SIZE, 0, strides);
    // int index_X = index(blockIdx.x * BLOCK_X_SIZE + BLOCK_X_SIZE, blockIdx.y * BLOCK_Y_SIZE, 0, strides);
    // int index_XY = index(blockIdx.x * BLOCK_X_SIZE + BLOCK_X_SIZE, blockIdx.y * BLOCK_Y_SIZE + BLOCK_Y_SIZE, 0, strides);

// flx and fly can be defined with smaller cache sizes, however in order to reuse the same cache_index function, I
// defined them here
// with same size. shared memory pressure should not be too high nevertheless
#define CACHE_SIZE (BLOCK_X_SIZE + HALO_BLOCK_X_MINUS + HALO_BLOCK_X_PLUS) * (BLOCK_Y_SIZE + 2)
#define CACHE_SIZE_IN (BLOCK_X_SIZE + 4) * (BLOCK_Y_SIZE + 4)

    __shared__ Real in_s[CACHE_SIZE_IN];
    __shared__ Real lap[CACHE_SIZE_IN];
    __shared__ Real flx[CACHE_SIZE_IN];
    __shared__ Real fly[CACHE_SIZE_IN];
    int acc=-1;

    for (int kpos = 0; kpos < domain.m_k; ++kpos) {

        if(threadIdx.x < 32 && threadIdx.y < 8 && threadIdx.x>=0 && threadIdx.y>=0){

            in_s[threadIdx.x + 36*threadIdx.y] = __ldg(& in[index(ipos2 , jpos2, kpos, strides)]);
            // printf("[%d, %d] %f = %f \n",  threadIdx.x, threadIdx.y, in_s[threadIdx.x + 36*threadIdx.y], __ldg(& in[threadIdx.x + threadIdx.y*36]));
        }

        if(threadIdx.x < 4 /*BLOCK_X_SIZE+halos - 32*/)
        {
            acc=cache_index_in(BLOCK_X_SIZE+threadIdx.x, threadIdx.y);
            in_s[acc] = __ldg(& in[ index( BLOCK_X_SIZE+ipos2, jpos2, kpos, strides)]);
        }

        if(threadIdx.y<4/*4*/ /*BLOCK_Y_SIZE+halos - 8*/)
        {
            acc=cache_index_in(threadIdx.x, threadIdx.y+BLOCK_Y_SIZE);
            // printf("address: %d\n", index(ipos2, BLOCK_Y_SIZE+jpos2, kpos, strides));
            in_s[acc] = __ldg(& in[index(ipos2, BLOCK_Y_SIZE+jpos2, kpos, strides)]);
            // acc=cache_index_in(threadIdx.x, threadIdx.y+BLOCK_Y_SIZE);
            // in_s[acc] = __ldg(& in[index_base + index( ipos2, BLOCK_Y_SIZE+jpos2, kpos, strides)]);
        }

        if(threadIdx.x<4 && threadIdx.y<4 )//&& BLOCK_X_SIZE+threadIdx.x+ipos2 < domain.m_i && BLOCK_Y_SIZE+jpos2+threadIdx.y < domain.m_j)
        {
            acc=cache_index_in(threadIdx.x+BLOCK_X_SIZE, threadIdx.y+BLOCK_Y_SIZE);
            in_s[acc] = __ldg(& in[ index(BLOCK_X_SIZE+ipos2, BLOCK_Y_SIZE+jpos2, kpos, strides)]);
        }

        __syncthreads();

        // if(threadIdx.x==0 && threadIdx.y==0){
        //     for(int i=0; i<36; ++i)
        //         for(int j=0; j<12; ++j)
        //             if(in_s[i+j*36] != __ldg(& in[ i + j * 36]))
        //                 printf("[%d,%d],%f = %f \n", i,j, in_s[i+j*36], __ldg(& in[ i + j * 36]));
        // }

        if(threadIdx.x < BLOCK_X_SIZE && threadIdx.y < BLOCK_Y_SIZE && threadIdx.x>=0 && threadIdx.y>=0){

            acc = cache_index_in(threadIdx.x+1, threadIdx.y+1);
            lap[acc] =
                (Real)4 * in_s[acc]
                -
                (  in_s[acc+1]
                   + in_s[acc-1] +
                   in_s[acc+36] + in_s[acc-36]);
        }

        // bool halo1=ipos2>0 && jpos2>0 && ipos2 < domain.m_i-1 && jpos2 < domain.m_j-1;
        if(threadIdx.x < 2 /*BLOCK_X_SIZE+halos - 32*/ // && halo1
            )
        {

            acc = threadIdx.x+1+BLOCK_X_SIZE + (threadIdx.y+1) * 36;
            lap[acc] =
                (Real)4 * in_s[acc]
                -
                (  in_s[acc+1]
                   + in_s[acc-1] +
                   in_s[acc+36] + in_s[acc-36]);
        }

        if(threadIdx.y < 2 /*BLOCK_X_SIZE+halos - 32*/ // && halo1
            )
        {

            acc = threadIdx.x+1 + (threadIdx.y+1+BLOCK_Y_SIZE)*36;
            lap[acc] =
                (Real)4 * in_s[acc]
                -
                (  in_s[acc+1]
                   + in_s[acc-1] +
                   in_s[acc+36] + in_s[acc-36]);
        }


        if(threadIdx.y < 2 && threadIdx.x < 2 // && halo1
            )
        {

            acc = (threadIdx.x+1+BLOCK_X_SIZE) + (threadIdx.y+1+BLOCK_Y_SIZE)*36;
            lap[acc] =
                (Real)4 * in_s[acc]
                -
                (  in_s[acc+1]
                   + in_s[acc-1] +
                   in_s[acc+36] + in_s[acc-36]);
        }

        __syncthreads();

        // if(threadIdx.x==0 && threadIdx.y==0){
        //     for(int i=1; i<35; ++i)
        //         for(int j=1; j<11; ++j)
        //             if(lap[i+j*36] != 4*in_s[i+j*36]-(in_s[(i+1)+j*36] + in_s[(i-1)+j*36] + in_s[i+(j+1)*36] + in_s[i+(j-1)*36]))
        //                printf("[%d,%d] %f = %f \n",i,j, lap[i+j*36], 4*in_s[i+j*36]-(in_s[(i+1)+j*36] + in_s[(i-1)+j*36] + in_s[i+(j+1)*36] + in_s[i+(j-1)*36]));
        // }


        if(threadIdx.x < BLOCK_X_SIZE && threadIdx.y < BLOCK_Y_SIZE && threadIdx.x>=0 && threadIdx.y>=0){

            acc = threadIdx.x+1 + (threadIdx.y+1)*36;
            flx[acc] =
                lap[acc+1] - lap[acc];
            if (flx[acc] *
                (in_s[acc+1] - in_s[acc]) > 0) {
                flx[acc] = 0.;
            }
        }

        if(threadIdx.x < 1 /*BLOCK_X_SIZE+halos*/)
        {
            acc = threadIdx.x+1+BLOCK_X_SIZE + (threadIdx.y+1) * 36;
            flx[acc] =
                lap[acc+1] - lap[acc];
            if (flx[acc] *
                (in_s[acc+1] - in_s[acc]) > 0) {
                flx[acc] = 0.;
            }
        }

        if(threadIdx.y < 2 /*BLOCK_X_SIZE+halos - 32*/ // && halo1
            )
        {

            acc = threadIdx.x+1 + (threadIdx.y+1+BLOCK_Y_SIZE)*36;
            flx[acc] =
                lap[acc+1] - lap[acc];
            if (flx[acc] *
                (in_s[acc+1] - in_s[acc]) > 0) {
                flx[acc] = 0.;
            }
        }


        if(threadIdx.y < 2 && threadIdx.x < 1 ) /*BLOCK_X_SIZE+halos - 32*/ // && halo1
        {
            acc = (threadIdx.x+1+BLOCK_X_SIZE) + (threadIdx.y+1+BLOCK_Y_SIZE)*36;
            flx[acc] =
                lap[acc+1] - lap[acc];
            if (flx[acc] *
                (in_s[acc+1] - in_s[acc]) > 0) {
                flx[acc] = 0.;
            }
        }


        //// FLUX Y

        if(threadIdx.x < BLOCK_X_SIZE && threadIdx.y < BLOCK_Y_SIZE && threadIdx.x>=0 && threadIdx.y>=0){

            acc = threadIdx.x+1 + (threadIdx.y+1)*36;
            fly[acc] =
                lap[acc+36] - lap[acc];
            if (fly[acc] *
                (in_s[acc+36] - in_s[acc]) > 0) {
                fly[acc] = 0.;
            }
        }

        if(threadIdx.x < 2 /*BLOCK_X_SIZE+halos*/)
        {
            acc = threadIdx.x+1+BLOCK_X_SIZE + (threadIdx.y+1) * 36;
            fly[acc] =
                lap[acc+36] - lap[acc];
            if (fly[acc] *
                (in_s[acc+36] - in_s[acc]) > 0) {
                fly[acc] = 0.;
            }
        }

        if(threadIdx.y < 1 /*BLOCK_X_SIZE+halos - 32*/ // && halo1
            )
        {

            acc = threadIdx.x+1 + (threadIdx.y+1+BLOCK_Y_SIZE)*36;
            fly[acc] =
                lap[acc+36] - lap[acc];
            if (fly[acc] *
                (in_s[acc+36] - in_s[acc]) > 0) {
                fly[acc] = 0.;
            }
        }


        if(threadIdx.y < 1 && threadIdx.x < 2 // && halo1
            )
        {
            acc = (threadIdx.x+1+BLOCK_X_SIZE) + (threadIdx.y+1+BLOCK_Y_SIZE)*36;
            fly[acc] =
                lap[acc+36] - lap[acc];
            if (fly[acc] *
                (in_s[acc+36] - in_s[acc]) > 0) {
                fly[acc] = 0.;
            }
        }

        __syncthreads();

        if(threadIdx.x < BLOCK_X_SIZE && threadIdx.y < BLOCK_Y_SIZE && threadIdx.x>=0 && threadIdx.y>=0){
            acc = (threadIdx.x+2) + (threadIdx.y+2)*36;
            out[index(ipos2+2, jpos2+2, kpos, strides)] =
                in_s[acc] -
                coeff[index(ipos2+2, jpos2+2, kpos, strides)] *
                (flx[acc] - flx[acc-1] +
                 fly[acc] - fly[acc-36]);
        }


        // if (is_in_domain< -1, 1, -1, 1 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {

        //     // printf("cache_index_in(%d, %d) = %d\n", threadIdx.x, threadIdx.y, cache_index_in(threadIdx.x, threadIdx.y));
        //     lap[cache_index(iblock_pos, jblock_pos)] =
        //         (Real)4 * in_s[acc-cache_index_in(1,1)]
        //         -
        //         (  __ldg(& in[index_ + index(1, 0, 0, strides)] )
        //            + __ldg(& in[index_ - index(1, 0,0, strides)] ) +
        //             __ldg(&in[index_+index(0, 1, 0, strides)]) + __ldg(&in[index_ - index(0, 1, 0, strides)]));
        // }

        // __syncthreads();

        // if (is_in_domain< -1, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
        //     flx[cache_index(iblock_pos, jblock_pos)] =
        //         lap[cache_index(iblock_pos + 1, jblock_pos)] - lap[cache_index(iblock_pos, jblock_pos)];
        //     if (flx[cache_index(iblock_pos, jblock_pos)] *
        //             (in_s[acc-cache_index_in(1,2)] - in_s[acc-cache_index_in(2,2)]) >
        //         0) {
        //         flx[cache_index(iblock_pos, jblock_pos)] = 0.;
        //     }
        // }

        // if (is_in_domain< 0, 0, -1, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
        //     fly[cache_index(iblock_pos, jblock_pos)] =
        //         lap[cache_index(iblock_pos, jblock_pos + 1)] - lap[cache_index(iblock_pos, jblock_pos)];
        //     if (fly[cache_index(iblock_pos, jblock_pos)] *
        //         (in_s[acc-cache_index_in(2,1)] - in_s[acc-cache_index_in(2,2)]) >
        //         0) {
        //         fly[cache_index(iblock_pos, jblock_pos)] = 0.;
        //     }
        // }

        // __syncthreads();

        // if (is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
        //     out[index_] =
        //          in_s[acc-cache_index_in(2,2)] -
        //         coeff[index_] *
        //             (flx[cache_index(iblock_pos, jblock_pos)] - flx[cache_index(iblock_pos - 1, jblock_pos)] +
        //                 fly[cache_index(iblock_pos, jblock_pos)] - fly[cache_index(iblock_pos, jblock_pos - 1)]);
        // }
        // index_ += index(0,0,1, strides);

        __syncthreads();
        index_base += index(0,0,1, strides);
    }
}

void launch_kernel(repository &repo, timer_cuda* time) {
    IJKSize domain = repo.domain();
    IJKSize halo = repo.halo();

    dim3 threads, blocks;
    threads.x = BLOCK_X_SIZE;
    threads.y = BLOCK_Y_SIZE;// + HALO_BLOCK_Y_MINUS + HALO_BLOCK_Y_PLUS + (HALO_BLOCK_X_MINUS > 0 ? 1 : 0) +
    //(HALO_BLOCK_X_PLUS > 0 ? 1 : 0);
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
