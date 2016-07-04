#pragma once
#include "../Definitions.hpp"
#include "../functions.hpp"

GT_FUNCTION void backward_sweep(const unsigned int i,
    const unsigned int j,
    const int iblock_pos,
    const int jblock_pos,
    const unsigned int block_size_i,
    const unsigned int block_size_j,
    Real *ccol,
    Real *dcol,
    Real *datacol,
    Real *u_pos,
    Real *utens_stage_ref,
    IJKSize const &domain,
    IJKSize const &strides) {
    // k maximum
    int k = domain.m_k - 1;
    if (is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {

        datacol[index(i, j, k, strides)] = dcol[index(i, j, k, strides)];
        ccol[index(i, j, k, strides)] = datacol[index(i, j, k, strides)];
        utens_stage_ref[index(i, j, k, strides)] =
            DTR_STAGE * (datacol[index(i, j, k, strides)] - u_pos[index(i, j, k, strides)]);
        // kbody
        for (k = domain.m_k - 2; k >= 0; --k) {
            datacol[index(i, j, k, strides)] =
                dcol[index(i, j, k, strides)] - (ccol[index(i, j, k, strides)] * datacol[index(i, j, k + 1, strides)]);
            ccol[index(i, j, k, strides)] = datacol[index(i, j, k, strides)];
            utens_stage_ref[index(i, j, k, strides)] =
                DTR_STAGE * (datacol[index(i, j, k, strides)] - u_pos[index(i, j, k, strides)]);
        }
    }
}

GT_FUNCTION void forward_sweep(const unsigned int i,
    const unsigned int j,
    const int iblock_pos,
    const int jblock_pos,
    const unsigned int block_size_i,
    const unsigned int block_size_j,
    const int ishift,
    const int jshift,
    Real *ccol,
    Real *dcol,
    Real *wcon,
    Real *u_stage,
    Real *u_pos,
    Real *utens,
    Real *utens_stage_ref,
    IJKSize const &domain,
    IJKSize const &strides) {

    // k minimum
    int k = 0;

    if (is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {

        Real gcv =
            (Real)0.25 * (wcon[index(i + ishift, j + jshift, k + 1, strides)] + wcon[index(i, j, k + 1, strides)]);
        Real cs = gcv * BET_M;

        ccol[index(i, j, k, strides)] = gcv * BET_P;
        Real bcol = DTR_STAGE - ccol[index(i, j, k, strides)];

        // update the d column
        Real correctionTerm = -cs * (u_stage[index(i, j, k + 1, strides)] - u_stage[index(i, j, k, strides)]);
        dcol[index(i, j, k, strides)] = DTR_STAGE * u_pos[index(i, j, k, strides)] + utens[index(i, j, k, strides)] +
                                        utens_stage_ref[index(i, j, k, strides)] + correctionTerm;

        Real divided = (Real)1.0 / bcol;
        ccol[index(i, j, k, strides)] = ccol[index(i, j, k, strides)] * divided;
        dcol[index(i, j, k, strides)] = dcol[index(i, j, k, strides)] * divided;
    }

    // kbody
    for (k = 1; k < domain.m_k - 1; ++k) {
        if (is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {

            Real gav = (Real)-0.25 * (wcon[index(i + ishift, j + jshift, k, strides)] + wcon[index(i, j, k, strides)]);
            Real gcv =
                (Real)0.25 * (wcon[index(i + ishift, j + jshift, k + 1, strides)] + wcon[index(i, j, k + 1, strides)]);

            Real as = gav * BET_M;
            Real cs = gcv * BET_M;

            Real acol = gav * BET_P;
            ccol[index(i, j, k, strides)] = gcv * BET_P;
            Real bcol = DTR_STAGE - acol - ccol[index(i, j, k, strides)];

            Real correctionTerm = -as * (u_stage[index(i, j, k - 1, strides)] - u_stage[index(i, j, k, strides)]) -
                                  cs * (u_stage[index(i, j, k + 1, strides)] - u_stage[index(i, j, k, strides)]);
            dcol[index(i, j, k, strides)] = DTR_STAGE * u_pos[index(i, j, k, strides)] +
                                            utens[index(i, j, k, strides)] + utens_stage_ref[index(i, j, k, strides)] +
                                            correctionTerm;

            Real divided = (Real)1.0 / (bcol - (ccol[index(i, j, k - 1, strides)] * acol));
            ccol[index(i, j, k, strides)] = ccol[index(i, j, k, strides)] * divided;
            dcol[index(i, j, k, strides)] =
                (dcol[index(i, j, k, strides)] - (dcol[index(i, j, k - 1, strides)] * acol)) * divided;
        }
    }

    // k maximum
    k = domain.m_k - 1;
    if (is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {

        Real gav = -(Real)0.25 * (wcon[index(i + ishift, j + jshift, k, strides)] + wcon[index(i, j, k, strides)]);
        Real as = gav * BET_M;

        Real acol = gav * BET_P;
        Real bcol = DTR_STAGE - acol;

        // update the d column
        Real correctionTerm = -as * (u_stage[index(i, j, k - 1, strides)] - u_stage[index(i, j, k, strides)]);
        dcol[index(i, j, k, strides)] = DTR_STAGE * u_pos[index(i, j, k, strides)] + utens[index(i, j, k, strides)] +
                                        utens_stage_ref[index(i, j, k, strides)] + correctionTerm;

        Real divided = (Real)1.0 / (bcol - (ccol[index(i, j, k - 1, strides)] * acol));
        dcol[index(i, j, k, strides)] =
            (dcol[index(i, j, k, strides)] - (dcol[index(i, j, k - 1, strides)] * acol)) * divided;
    }
}
