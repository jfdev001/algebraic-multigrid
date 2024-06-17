#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace AMG {

/**
 * @brief Return residual sum of squares of `bhat` from `Au` and rhs `b`.
 * 
 * @tparam EleType 
 * @param A Coefficients matrix for discretized governing equations. 
 * @param u Solution to linear system of equations. 
 * @param b Right hand side of linear system `Au = b`. 
 * @return EleType 
 */
template <class EleType>
EleType rss(const Eigen::SparseMatrix<EleType>& A,
            const Eigen::Matrix<EleType, -1, 1>& u,
            const Eigen::Matrix<EleType, -1, 1>& b) {
  auto bhat = A * u;
  EleType error = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    error += (b[i] - bhat[i]) * (b[i] - bhat[i]);
  }
  return error;
}

}  // end namespace AMG