#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace AMG {

/**
 * @brief Return residual sum of squares of `bhat` from `Au` and rhs `b`.
 * 
 * @tparam EleType 
 * @return double 
 */
template <class EleType>
double residual(
    const Eigen::SparseMatrix<EleType>& A,
    const Eigen::Matrix<EleType, -1, 1>& u,
    const Eigen::Matrix<EleType, -1, 1>& b) {
    auto bhat = A*u; 
    double error = 0;
    for (size_t i = 0; i < b.size(); ++i) {
        error += pow(b[i] - bhat[i], 2);
    }
    return error;
}

} // end namespace AMG