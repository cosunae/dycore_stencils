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
        //ccol[index(i, j, k, strides)] = datacol[index(i, j, k, strides)];
        utens_stage_ref[index(i, j, k, strides)] =
            DTR_STAGE * (datacol[index(i, j, k, strides)] - u_pos[index(i, j, k, strides)]);
        // kbody
        for (k = domain.m_k - 2; k >= 0; --k) {
            datacol[index(i, j, k, strides)] =
                dcol[index(i, j, k, strides)] - (ccol[index(i, j, k, strides)] * datacol[index(i, j, k + 1, strides)]);
            //ccol[index(i, j, k, strides)] = datacol[index(i, j, k, strides)];
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

#ifdef USE_CACHE_
    Real ccol_cache[2];
    Real dcol_cache[2];
#endif

    if (is_in_domain< 0, 0, 0, 0 >(iblock_pos, jblock_pos, block_size_i, block_size_j)) {

        Real gcv =
            (Real)0.25 * (wcon[index(i + ishift, j + jshift, k + 1, strides)] + wcon[index(i, j, k + 1, strides)]);
        Real cs = gcv * BET_M;

#ifdef USE_CACHE_
        ccol_cache[0] = gcv * BET_P;
        Real bcol = DTR_STAGE - ccol_cache[0];
#else
        ccol[index(i, j, k, strides)] = gcv * BET_P;
        Real bcol = DTR_STAGE - ccol[index(i, j, k, strides)];
#endif
        // update the d column
        Real correctionTerm = -cs * (u_stage[index(i, j, k + 1, strides)] - u_stage[index(i, j, k, strides)]);
#ifdef USE_CACHE_
        dcol_cache[0]
#else
        dcol[index(i, j, k, strides)]
#endif
            = DTR_STAGE * u_pos[index(i, j, k, strides)] + utens[index(i, j, k, strides)] +
                                        utens_stage_ref[index(i, j, k, strides)] + correctionTerm;

        Real divided = (Real)1.0 / bcol;
#ifdef USE_CACHE_
        ccol_cache[0] = ccol_cache[0] * divided;
        ccol[index(i, j, k + 1, strides)] = ccol_cache[0];
        dcol_cache[0] *= divided;
        dcol[index(i, j, k + 1, strides)] = dcol_cache[0];
#else
        ccol[index(i, j, k, strides)] = ccol[index(i, j, k, strides)] * divided;
        dcol[index(i, j, k, strides)] = dcol[index(i, j, k, strides)] * divided;
#endif
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
#ifdef USE_CACHE_
            ccol_cache[1] = gcv * BET_P;
            Real bcol = DTR_STAGE - acol - ccol_cache[1];
#else
            ccol[index(i, j, k, strides)] = gcv * BET_P;
            Real bcol = DTR_STAGE - acol - ccol[index(i, j, k, strides)];
#endif

            Real correctionTerm = -as * (u_stage[index(i, j, k - 1, strides)] - u_stage[index(i, j, k, strides)]) -
                                  cs * (u_stage[index(i, j, k + 1, strides)] - u_stage[index(i, j, k, strides)]);
#ifdef USE_CACHE_
            dcol_cache[1]
#else
            dcol[index(i, j, k, strides)]
#endif
                = DTR_STAGE * u_pos[index(i, j, k, strides)] +
                utens[index(i, j, k, strides)] + utens_stage_ref[index(i, j, k, strides)] +
                correctionTerm;

#ifdef USE_CACHE_
            Real divided = (Real)1.0 / (bcol - (ccol_cache[0] * acol));
            ccol_cache[1] = ccol_cache[1] * divided;

            ccol_cache[1] = ccol_cache[1] * divided;
            ccol[index(i, j, k + 1, strides)] = ccol_cache[0];
            ccol_cache[0]=ccol_cache[1];

            dcol_cache[1] = (dcol_cache[1] - (dcol_cache[0] * acol)) * divided;
            dcol[index(i, j, k + 1, strides)] = dcol_cache[0];
            dcol_cache[0]=dcol_cache[1];
#else
            Real divided = (Real)1.0 / (bcol - (ccol[index(i, j, k - 1, strides)] * acol));
            ccol[index(i, j, k, strides)] = ccol[index(i, j, k, strides)] * divided;
            dcol[index(i, j, k, strides)] =
                (dcol[index(i, j, k, strides)] - (dcol[index(i, j, k - 1, strides)] * acol)) * divided;
#endif
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

#ifdef USE_CACHE_
        Real divided = (Real)1.0 / (bcol - (ccol_cache[0] * acol));
#else
        Real divided = (Real)1.0 / (bcol - (ccol[index(i, j, k - 1, strides)] * acol));
#endif
        dcol[index(i, j, k, strides)] =
            (dcol[index(i, j, k, strides)] - (dcol[index(i, j, k - 1, strides)] * acol)) * divided;
    }
}
