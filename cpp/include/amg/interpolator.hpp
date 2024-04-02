#pragma once

#include <Eigen/Sparse>
#include <vector>

namespace AMG {

/**
 * @brief Interface for restriction and prolongation operators.
 * 
 * References:
 * 
 * [1] : PyAMG. url: https://github.com/pyamg/pyamg/blob/main/pyamg/strength.py
 * [2] : AlgebraicMultigrid.jl. url: https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/blob/master/src/strength.jl
 *  and url: https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/blob/master/src/classical.jl
 */
template <class EleType>
class Interpolator {
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
  Interpolator() {}
  Interpolator(EleType theta_) : theta(theta_) {
    if (theta_ < 0 || theta_ > 1) {
      std::string msg =
          "theta must be in [0, 1], but got " + std::to_string(theta);
      throw(std::invalid_argument(msg));
    }
  }
  ~Interpolator() = default;
};

} // end namespace AMG 