#pragma once

#include <Eigen/Sparse>
#include <vector>

namespace AMG {

template <class EleType>
class InterpolatorBase {
 private:
  std::vector<Eigen::SparseMatrix<EleType>> level_to_P;
  std::vector<Eigen::SparseMatrix<EleType>> level_to_R;

 public:
  virtual void prolongation(const Eigen::Matrix<EleType, -1, 1>& v,
                            size_t level) = 0;
  virtual void restriction(const Eigen::Matrix<EleType, -1, 1>& v,
                           size_t level) = 0;
};

/**
 * @brief Interface for linear interpolation.
 * 
 * References:
 * 
 * [1] : 
 * 
 * @tparam EleType 
 */
template <class EleType>
class LinearInterpolator : public InterpolatorBase<EleType> {
 private:
  const size_t n_elements_per_columns = 3;

 public:
  LinearInterpolator(std::vector<size_t> level_to_ndofs) {
    size_t n_fine_dofs;
    size_t n_coarse_dofs;
    for (size_t level = 0; level < level_to_ndofs.size() - 1; ++level) {
      n_fine_dofs = level_to_ndofs[level];
      n_coarse_dofs = level_to_ndofs[level + 1];

      // Create prolongation matrix
      size_t nnz = n_coarse_dofs * n_elements_per_columns;
      Eigen::SparseMatrix<EleType> P(n_fine_dofs, n_coarse_dofs);
      P.reserve(nnz);

      // Populate nonzeros in matrix
      std::vector<Eigen::Triplet<EleType>> P_coefficients;
      P_coefficients.reserve(nnz);
      size_t i = 0;
      for (size_t j = 0; j < n_coarse_dofs; ++j) {
        P_coefficients.push_back(Eigen::Triplet<EleType>(i, j, 0.5));
        P_coefficients.push_back(Eigen::Triplet<EleType>(i + 1, j, 1.0));
        P_coefficients.push_back(Eigen::Triplet<EleType>(i + 2, j, 0.5));
        i += n_elements_per_columns - 1;
      }
      P.setFromTriplets(P_coefficients.begin(), P_coefficients.end());

      // // Restriction matrix follows from prolongation matrix
      // Eigen::SparseMatrix<EleType> R(n_coarse_dofs, n_fine_dofs);
      // R.reserve(P.nonZeros());
      // R = P.transpose();
    }
  }
  void prolongation(const Eigen::Matrix<EleType, -1, 1>& v, size_t level) {}
  void restriction(const Eigen::Matrix<EleType, -1, 1>& v, size_t level) {}
};

/**
 * @brief Interface for direct interpolation using classical strength.
 * 
 * References:
 * 
 * [1] : PyAMG. url: https://github.com/pyamg/pyamg/blob/main/pyamg/strength.py
 * [2] : AlgebraicMultigrid.jl. url: https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/blob/master/src/strength.jl
 *  and url: https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/blob/master/src/classical.jl
 */
template <class EleType>
class DirectInterpolator : public InterpolatorBase<EleType> {
 private:
  /**
   * @brief Threshold parameter.
   * 
   */
  EleType theta{0.25};

  /**
   * @brief Defines strength of connection matrices.
   * 
   * References:
   * 
   * [1] : https://github.com/pyamg/pyamg/blob/main/pyamg/strength.py
   * 
   * @param A 
   * @param S 
   * @param T 
   */
  void classical_strength(const Eigen::SparseMatrix<EleType>& A,
                          Eigen::SparseMatrix<EleType>& S,
                          Eigen::SparseMatrix<EleType>& T) {
    int n = A.cols();
    int row;
    int col;
    EleType _m;
    EleType threshold;
    EleType val;
    Eigen::SparseMatrix<EleType> strength{A};
    for (int i = 0; i < n; ++i) {
      _m = find_max_off_diag(A, i);
      threshold = theta * _m;
      for (typename Eigen::SparseMatrix<EleType>::Iterator it(A, i); it; ++it) {
        row = it.row();
        col = it.row();
        val = it.value();
        if (row != i) {
          if (std::abs(val) >= threshold) {
            // update
            strength.insert(row, col) = std::abs(val);
          } else {
            // set zero
            strength.insert(row, col) = std::abs(val);
          }
        }
      }
    }

    auto nnz_prev = strength.nonZeros();
    strength.prune(0.0);  // inplace?
    auto nnz_post = T.nonZeros();
    // TODO: remove
    std::cout << nnz_prev - nnz_post << " <-- dropped zeros" << std::endl;

    scale_cols_by_largest_entry(T);

    return;
  }

  EleType find_max_off_diag(const Eigen::SparseMatrix<EleType>& A, int i) {
    EleType m = 0;
    int row;
    EleType val;
    for (typename Eigen::SparseMatrix<EleType>::Iterator it(A, i); it; ++it) {
      row = it.row();
      val = it.value();
      if (row != i) {
        m = std::max(m, std::abs(val));
      }
    }
    return m;
  }

  EleType find_max(const Eigen::SparseMatrix<EleType>& A, int i) {
    EleType m;
    int row;
    EleType val;
    for (typename Eigen::SparseMatrix<EleType>::Iterator it(A, i); it; ++it) {
      row = it.row();
      val = it.value();
      m = std::max(m, val);
    }
    return m;
  }

  void scale_cols_by_largest_entry(Eigen::SparseMatrix<EleType>& A) {
    EleType _m;
    int row;
    int col;
    for (int i = 0; i < A.outerSize(); ++i) {
      _m = find_max(A, i);
      for (typename Eigen::SparseMatrix<EleType>::Iterator it(A, i); it; ++it) {
        row = it.row();
        col = it.col();
        A.insert(row, col) = A.coeff(row, col) / _m;
      }
    }
    return;
  }

 public:
  DirectInterpolator() { throw(std::logic_error("notimplemented")); }
  DirectInterpolator(EleType theta_) : theta(theta_) {
    throw(std::logic_error("notimplemented"));
    if (theta_ < 0 || theta_ > 1) {
      std::string msg =
          "theta must be in [0, 1], but got " + std::to_string(theta);
      throw(std::invalid_argument(msg));
    }
  }
  ~DirectInterpolator() = default;

  void prolongation(const Eigen::Matrix<EleType, -1, 1>& v, size_t level) {}
  void restriction(const Eigen::Matrix<EleType, -1, 1>& v, size_t level) {}
};

}  // end namespace AMG