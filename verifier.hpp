#pragma once
#include "utils.hpp"

template < typename value_type >
bool compare_below_threshold(value_type expected, value_type actual, double precision) {
    if (std::fabs(expected) < 1e-3 && std::fabs(actual) < 1e-3) {
        if (std::fabs(expected - actual) < precision)
            return true;
    } else {
        if (std::fabs((expected - actual) / (precision * expected)) < 1.0)
            return true;
    }
    return false;
}


struct verifier {

    IJKSize m_domain;
    IJKSize m_halo;
    IJKSize m_strides;
    double m_precision;
    verifier(IJKSize domain, IJKSize halo, double precision) : m_domain(domain), m_halo(halo), m_precision(precision) {
        compute_strides(m_domain, halo, m_strides);
    }

    bool verify(Real* expected_field, Real* actual_field)
    {
        bool verified = true;
        for (unsigned int k = 0; k < m_domain.m_k; ++k) {
            for (unsigned int i = m_halo.m_i; i < m_domain.m_i - m_halo.m_i; ++i) {
                for (unsigned int j = m_halo.m_j; j < m_domain.m_j - m_halo.m_j; ++j) {
                    Real expected = expected_field[index(i,j,k, m_strides)];
                    Real actual = actual_field[index(i,j,k,m_strides)];

                    if (!compare_below_threshold(expected, actual, m_precision)) {
                        std::cout << "Error in position " << i << " " << j << " " << k
                                  << " ; expected : " << expected << " ; actual : " << actual << "  "
                                  << std::fabs((expected - actual) / (expected)) << std::endl;
                        verified=false;
                    }
                }
            }
        }
        return verified;
    }
};
