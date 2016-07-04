#pragma once

#ifndef PADDING_LEFT
WELL_NEED_TO_DEFINE_IT
#endif

#ifndef BLOCK_X_SIZE
WELL_NEED_TO_DEFINE_IT
#endif

inline unsigned int padded_size_right(IJKSize const &domain) {
    const unsigned int blocks_x = (domain.m_i + BLOCK_X_SIZE - 1) / BLOCK_X_SIZE;

    return blocks_x * BLOCK_X_SIZE;
}

inline unsigned int padded_size(IJKSize const &domain) { return PADDING_LEFT + padded_size_right(domain); }

inline void compute_strides(IJKSize const &domain, IJKSize const& halo, IJKSize& strides) {

    strides.m_i = 1;
    strides.m_j = strides.m_i * padded_size_right(domain);
    strides.m_k = strides.m_j * domain.m_j;
}

GT_FUNCTION
unsigned int index(const unsigned int ipos, const unsigned int jpos, const unsigned int kpos, IJKSize const &strides) {
    return PADDING_LEFT + ipos * strides.m_i + jpos * strides.m_j + kpos * strides.m_k;
}
