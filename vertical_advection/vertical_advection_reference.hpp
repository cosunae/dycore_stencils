#pragma once
#include "../utils.hpp"
#include "functions.hpp"

struct vertical_advection_reference {

    vertical_advection_reference(repository &repo) : m_repo(repo), m_domain(repo.domain()), m_halo(repo.halo()) {
        compute_strides(m_domain, m_halo, m_strides);
    }

    void generate_reference() {
        m_repo.init_field("ccol", -1.0);
        m_repo.init_field("dcol", -1.0);
        m_repo.init_field("datacol", -1.0);

        Real *ccol = m_repo.field_h("ccol");
        Real *dcol = m_repo.field_h("dcol");
        Real *datacol = m_repo.field_h("datacol");

        u_stage_ = m_repo.field_h("u_stage");
        wcon_ = m_repo.field_h("wcon");
        u_pos_ = m_repo.field_h("u_pos");
        utens_ = m_repo.field_h("utens");
        utens_stage_ref_ = m_repo.field_h("utens_stage_ref");

        v_stage_ = m_repo.field_h("v_stage");
        v_pos_ = m_repo.field_h("v_pos");
        vtens_ = m_repo.field_h("vtens");
        vtens_stage_ref_ = m_repo.field_h("vtens_stage_ref");

        w_stage_ = m_repo.field_h("w_stage");
        w_pos_ = m_repo.field_h("w_pos");
        wtens_ = m_repo.field_h("wtens");
        wtens_stage_ref_ = m_repo.field_h("wtens_stage_ref");

        // Generate U
        for (int i = m_halo.m_i; i < m_domain.m_i - m_halo.m_i; ++i) {
            for (int j = m_halo.m_j; j < m_domain.m_j - m_halo.m_j; ++j) {

                forward_sweep(
                    i, j, 0,0, 8,8, 1, 0, ccol, dcol, wcon_, u_stage_, u_pos_, utens_, utens_stage_ref_, m_domain, m_strides);
                backward_sweep(i, j, 0,0,8,8,ccol, dcol, datacol, u_pos_, utens_stage_ref_, m_domain, m_strides);
            }
        }

        // Generate V
        for (int i = m_halo.m_i; i < m_domain.m_i - m_halo.m_i; ++i) {
            for (int j = m_halo.m_j; j < m_domain.m_j - m_halo.m_j; ++j) {

                forward_sweep(
                    i, j, 0,0, 8,8, 1, 0, ccol, dcol, wcon_, v_stage_, v_pos_, vtens_, vtens_stage_ref_, m_domain, m_strides);
                backward_sweep(i, j, 0,0,8,8,ccol, dcol, datacol, v_pos_, vtens_stage_ref_, m_domain, m_strides);
            }
        }

        // Generate W
        for (int i = m_halo.m_i; i < m_domain.m_i - m_halo.m_i; ++i) {
            for (int j = m_halo.m_j; j < m_domain.m_j - m_halo.m_j; ++j) {

                forward_sweep(
                    i, j, 0,0, 8,8, 1, 0, ccol, dcol, wcon_, w_stage_, w_pos_, wtens_, wtens_stage_ref_, m_domain, m_strides);
                backward_sweep(i, j, 0,0,8,8,ccol, dcol, datacol, w_pos_, wtens_stage_ref_, m_domain, m_strides);
            }
        }

    }

  private:
    repository m_repo;
    IJKSize m_domain;
    IJKSize m_halo;
    IJKSize m_strides;
    Real *utens_stage_, *u_stage_, *wcon_, *u_pos_, *utens_, *utens_stage_ref_;
    Real *vtens_stage_, *v_stage_, *v_pos_, *vtens_, *vtens_stage_ref_;
    Real *wtens_stage_, *w_stage_, *w_pos_, *wtens_, *wtens_stage_ref_;

};
