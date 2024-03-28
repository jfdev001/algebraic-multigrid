#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace AMG {

/**
 * @brief Return residual sum of squares of `bhat` from `Au` and rhs `b`.
 * 
 * @tparam EleType 
 * @param A 
 * @param u 
 * @param b 
 * @return EleType 
 */
template <class EleType>
EleType residual(const Eigen::SparseMatrix<EleType>& A,
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