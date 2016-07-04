#pragma once
#include "../utils.hpp"

struct horizontal_diffusion_reference {

    horizontal_diffusion_reference(repository &repo) : m_repo(repo) {}

    void generate_reference() {

        m_repo.make_field("u_diff_ref");
        m_repo.make_field("lap_ref");
        m_repo.make_field("flx_ref");
        m_repo.make_field("fly_ref");

        m_repo.init_field("u_diff_ref", 0.0);
        m_repo.init_field("lap_ref", 0.0);
        m_repo.init_field("flx_ref", 0.0);
        m_repo.init_field("fly_ref", 0.0);

        Real *u_in = m_repo.field_h("u_in");
        Real *u_diff_ref = m_repo.field_h("u_diff_ref");
        Real *lap = m_repo.field_h("lap_ref");
        Real *flx = m_repo.field_h("flx_ref");
        Real *fly = m_repo.field_h("fly_ref");
        Real *coeff = m_repo.field_h("coeff");

        IJKSize domain = m_repo.domain();
        IJKSize halo = m_repo.halo();
        IJKSize strides;

        compute_strides(domain, halo, strides);
        for (unsigned int k = 0; k < domain.m_k; ++k) {
            for (unsigned int i = halo.m_i - 1; i < domain.m_i - halo.m_i + 1; ++i) {
                for (unsigned int j = halo.m_j - 1; j < domain.m_j - halo.m_j + 1; ++j) {
                    lap[index(i, j, k, strides)] =
                        (Real)4 * u_in[index(i, j, k, strides)] -
                        (u_in[index(i + 1, j, k, strides)] + u_in[index(i, j + 1, k, strides)] +
                            u_in[index(i - 1, j, k, strides)] + u_in[index(i, j - 1, k, strides)]);
                }
            }
            for (unsigned int i = halo.m_i - 1; i < domain.m_i - halo.m_i; ++i) {
                for (unsigned int j = halo.m_j; j < domain.m_j - halo.m_j; ++j) {
                    flx[index(i, j, k, strides)] = lap[index(i + 1, j, k, strides)] - lap[index(i, j, k, strides)];
                    if (flx[index(i, j, k, strides)] *
                            (u_in[index(i + 1, j, k, strides)] - u_in[index(i, j, k, strides)]) >
                        0)
                        flx[index(i, j, k, strides)] = 0.;
                }
            }
            for (unsigned int i = halo.m_i; i < domain.m_i - halo.m_i; ++i) {
                for (unsigned int j = halo.m_j - 1; j < domain.m_j - halo.m_j; ++j) {
                    fly[index(i, j, k, strides)] = lap[index(i, j + 1, k, strides)] - lap[index(i, j, k, strides)];
                    if (fly[index(i, j, k, strides)] *
                            (u_in[index(i, j + 1, k, strides)] - u_in[index(i, j, k, strides)]) >
                        0)
                        fly[index(i, j, k, strides)] = 0.;
                }
            }
            for (unsigned int i = halo.m_i; i < domain.m_i - halo.m_i; ++i) {
                for (unsigned int j = halo.m_j; j < domain.m_j - halo.m_j; ++j) {
                    u_diff_ref[index(i, j, k, strides)] =
                        u_in[index(i, j, k, strides)] -
                        coeff[index(i, j, k, strides)] *
                            (flx[index(i, j, k, strides)] - flx[index(i - 1, j, k, strides)] +
                                fly[index(i, j, k, strides)] - fly[index(i, j - 1, k, strides)]);
                }
            }
        }
    }

  private:
    repository &m_repo;
};
