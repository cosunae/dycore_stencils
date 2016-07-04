#pragma once

inline void compute_strides(IJKSize const & domain, IJKSize const& halo, IJKSize & strides)
{
    strides.m_i = 1;
    strides.m_j = 132;
    strides.m_k = 132*132;
}

__host__ __device__
inline unsigned int index(const unsigned int ipos, const unsigned int jpos, const unsigned int kpos)
{
    return ipos*1 + jpos*132 + kpos*132*132;
}
__host__ __device__


inline unsigned int index(const unsigned int ipos, const unsigned int jpos, const unsigned int kpos, IJKSize const & strides)
{
    return ipos*strides.m_i + jpos*strides.m_j + kpos*strides.m_k;
}
