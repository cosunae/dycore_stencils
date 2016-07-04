#pragma once
#include <map>
#include <string>
#include <assert.h>
#include <cuda_runtime.h>
#include "domain.hpp"
#include "utils.hpp"

struct repository {

    repository(IJKSize domain, IJKSize halo)
        : m_domain(domain), m_halo(halo), m_field_size((domain.m_i+halo.m_i*2) * (domain.m_j+halo.m_j*2) * domain.m_k) {
        compute_strides(m_domain, halo, m_strides);
    }

    void make_field(std::string name) {
        Real *ptr = new Real[m_field_size];
        m_fields_h[name] = ptr;
        Real *ptr_d;
        cudaError_t error = cudaMalloc(&ptr_d, sizeof(Real) * m_field_size);
        m_fields_d[name] = ptr_d;
    }

    Real *field_h(std::string name) {
        assert(m_fields_h[name]);
        return m_fields_h[name];
    }
    Real *field_d(std::string name) {
        assert(m_fields_d[name]);
        return m_fields_d[name];
    }
    void update_host(std::string name) {
        cudaMemcpy(field_h(name), field_d(name), m_field_size * sizeof(Real), cudaMemcpyDeviceToHost);
    }

    void update_device(std::string name) {
        cudaMemcpy(field_d(name), field_h(name), m_field_size * sizeof(Real), cudaMemcpyHostToDevice);
    }

    void fill_field(std::string name, Real offset1, Real offset2, Real base1, Real base2, Real spreadx, Real spready) {
        const unsigned int i_begin = 0;
        const unsigned int i_end = m_domain.m_i+m_halo.m_i*2;
        const unsigned int j_begin = 0;
        const unsigned int j_end = m_domain.m_j+m_halo.m_j*2;
        const unsigned int k_begin = 0;
        const unsigned int k_end = m_domain.m_k;

        Real dx = 1. / (Real)(i_end - i_begin);
        Real dy = 1. / (Real)(j_end - j_begin);
        Real dz = 1. / (Real)(k_end - k_begin);

        Real *field = field_h(name);

        for (int j = j_begin; j < j_end; j++) {
            for (int i = i_begin; i < i_end; i++) {
                double x = dx * (double)(i - i_begin);
                double y = dy * (double)(j - j_begin);
                for (int k = k_begin; k < k_end; k++) {
                    double z = dz * (double)(k - k_begin);

                    // u values between 5 and 9
                    field[index(i, j, k, m_strides)] = offset1 +
                                                       base1 *
                                                           (offset2 + cos(PI * (spreadx * x + spready * y)) +
                                                               base2 * sin(2 * PI * (spreadx * x + spready * y) * z)) /
                                                           4.;
                }
            }
        }
    }

    void init_field(std::string name, Real value) {
        const unsigned int i_begin = 0;
        const unsigned int i_end = m_domain.m_i+m_halo.m_i*2;
        const unsigned int j_begin = 0;
        const unsigned int j_end = m_domain.m_j+m_halo.m_j*2;
        const unsigned int k_begin = 0;
        const unsigned int k_end = m_domain.m_k;

        Real *field = field_h(name);

        for (int j = j_begin; j < j_end; j++) {
            for (int i = i_begin; i < i_end; i++) {
                for (int k = k_begin; k < k_end; k++) {
                    // u values between 5 and 9
                    field[index(i, j, k, m_strides)] = value;
                }
            }
        }
    }

    IJKSize halo() { return m_halo; }
    IJKSize domain() { return m_domain; }

  private:
    const IJKSize m_domain;
    const IJKSize m_halo;
    IJKSize m_strides;
    const unsigned int m_field_size;
    std::map< std::string, Real * > m_fields_h;
    std::map< std::string, Real * > m_fields_d;
};
