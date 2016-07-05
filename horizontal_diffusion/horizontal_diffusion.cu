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
    double2 *in, double2 *out, double2 *coeff, const IJKSize domain, const IJKSize halo, const IJKSize strides) {

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
    __shared__ double2 lap[CACHE_SIZE];
    __shared__ double2 flx[CACHE_SIZE];
    __shared__ double2 fly[CACHE_SIZE];

    for (int kpos = 0; kpos < domain.m_k; ++kpos) {

        if (is_in_domain< -1, 1, -1, 1 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
    
            double2 in_jp1 = __ldg(&in[index_+index(0, 1, 0, strides)]);
            double2 in_jm1 = __ldg(&in[index_-index(0, 1, 0, strides)]);

            double2 in_c = __ldg(& in[index_] ); 
            if(iblock_pos > -1 && iblock_pos < block_size_i ) {
                double2 in_im1 = __ldg( (&in[index_ - index(1,0,0,strides)]));
                double2 lap_c;
                
                lap_c.x =
                     (double)4.0 * in_c.x -
                    ( in_c.y + in_im1.y +
                    in_jp1.x + in_jm1.x);

                double2 in_ip1 = __ldg(& in[index_+index(1, 0,0, strides)] );
                lap_c.y =
                    (double)4.0 * in_c.y -
                    ( in_c.x + in_ip1.x +
                     in_jp1.y + in_jm1.y);


                
                lap[cache_index(iblock_pos, jblock_pos)] = lap_c;
            }
            else if(iblock_pos > -1 ) {
                double2 in_im1 = __ldg( (&in[index_ - index(1,0,0,strides)]));
  
                lap[cache_index(iblock_pos, jblock_pos)].x =
                     (double)4.0 * in_c.x -
                    ( in_c.y + in_im1.y +
                    __ldg(&in[index_+index(0, 1, 0, strides)]).x + __ldg(&in[index_ - index(0, 1, 0, strides)]).x);

            }
            else if(iblock_pos < block_size_i ) {

                double2 in_ip1 = __ldg(& in[index_+index(1, 0,0, strides)] );
                lap[cache_index(iblock_pos, jblock_pos)].y =
                    (double)4.0 * in_c.y -
                    ( in_c.x + in_ip1.x +
                    __ldg(&in[index_+index(0, 1, 0, strides)]).y + __ldg(&in[index_ - index(0, 1, 0, strides)]).y);

            }
        }

        __syncthreads();

        if (is_in_domain< -1, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            double2 in_c = __ldg(& in[index_] );
            double2 lap_c = lap[cache_index(iblock_pos, jblock_pos)];
            if(iblock_pos > -1) {
                flx[cache_index(iblock_pos, jblock_pos)].x = 
                    lap_c.y - lap_c.x;
                if (flx[cache_index(iblock_pos, jblock_pos)].x *
                        (in_c.y - in_c.x) >
                    0) {
                    flx[cache_index(iblock_pos, jblock_pos)].x = 0.;
                }
            }

            double2 lap_ip1 = lap[cache_index(iblock_pos + 1, jblock_pos)];
            double2 in_ip1 = __ldg(& in[index_+index(1, 0,0, strides)] );
            double2 flx_c; 

            flx_c.y =
                lap_ip1.x - lap_c.y;
            if (flx_c.y *
                    (in_ip1.x - in_c.y) >
                0) {
                flx_c.y = 0.;
            }

            flx[cache_index(iblock_pos, jblock_pos)] = flx_c;
 
        }

        if (is_in_domain< 0, 0, -1, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            double2 fly_c;
            double2 lap_jp1 = lap[cache_index(iblock_pos, jblock_pos + 1)];
            double2 lap_c = lap[cache_index(iblock_pos, jblock_pos)];
            double2 in_jp1 = __ldg(&in[index_+index(0, 1, 0, strides)]);
            double2 in_c = __ldg(&in[index_]);

            fly_c.x =
                lap_jp1.x - lap_c.x;
            if (fly_c.x *
                    (in_jp1.x - in_c.x) >
                0) {
                fly_c.x = 0.;
            }
            fly_c.y =
                lap_jp1.y - lap_c.y;
            if (fly_c.y *
                    (in_jp1.y - in_c.y) >
                0) {
                fly_c.y = 0.;
            }

            fly[cache_index(iblock_pos, jblock_pos)] = fly_c;

 
        }

        __syncthreads();

        if (is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {
            double2 flx_c = flx[cache_index(iblock_pos, jblock_pos)];
            double2 flx_im1 = flx[cache_index(iblock_pos-1, jblock_pos)];
            double2 in_c = __ldg(&in[index_]);
            double2 coeff_c = coeff[index_];
            double2 fly_c  = fly[cache_index(iblock_pos, jblock_pos)];
            double2 fly_jm1 = fly[cache_index(iblock_pos, jblock_pos - 1)];

            out[index_].x = 
                  in_c.x -
                coeff_c.x *
                    (flx_c.x - flx_im1.y +
                        fly_c.x - fly_jm1.x);
            out[index_].y =
                in_c.y -
                coeff_c.y *
                    (flx_c.y - flx_c.x +
                        fly_c.y - fly_jm1.y);
//printf("PPPPP %d %d %d %f %f %p\n", ipos, jpos, kpos, out[index_].x, out[index_].y, &(out[index_].x)); 
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
    blocks.x = (domain.m_i + (BLOCK_X_SIZE*2) - 1) / (BLOCK_X_SIZE*2);
    blocks.y = (domain.m_j + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE;
    blocks.z = 1;

    IJKSize strides;
    IJKSize domain_d2 = domain;
    domain_d2.m_i /= 2;
    IJKSize halo_d2 = halo; 
    halo_d2.m_i /=2;
    compute_strides(domain_d2, halo_d2, strides);

    double2 *in = (double2*) repo.field_d("u_in");
    double2 *out = (double2*) repo.field_d("u_out");
    double2 *coeff = (double2*)repo.field_d("coeff");

    if(time) time->start();
    cukernel<<< blocks, threads, 0 >>>(in, out, coeff, domain, halo, strides);
    if(time) time->pause();
}
