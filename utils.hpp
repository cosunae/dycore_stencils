#pragma once

inline void compute_strides(IJKSize const & domain, IJKSize const& halo, IJKSize & strides)
{
    strides.m_i = 1;
    strides.m_j = strides.m_i * (domain.m_i + 2*halo.m_i);
    strides.m_k = strides.m_j * (domain.m_j + 2*halo.m_j);
}

__host__ __device__
inline unsigned int index(const unsigned int ipos, const unsigned int jpos, const unsigned int kpos, IJKSize const & strides)
{
    return ipos*strides.m_i + jpos*strides.m_j + kpos*strides.m_k;
}
