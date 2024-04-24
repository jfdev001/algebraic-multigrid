#pragma once

#include <Eigen/Sparse>
#include <vector>

namespace AMG {

/**
 * @brief Base class for interpolators that implement prolongation and 
 * restriction as linear operators constructed by the `make_operators`
 * virtual function.
 * 
 * @tparam EleType 
 */
template <class EleType>
class InterpolatorBase {
 private:
  std::vector<Eigen::SparseMatrix<EleType>> level_to_P;
  std::vector<Eigen::SparseMatrix<EleType>> level_to_R;

 public:
  InterpolatorBase(size_t n_levels) {
    // only need operators for levels 0 to n_levels-1
    level_to_P.resize(n_levels - 1);
    level_to_R.resize(n_levels - 1);
  }

  /**
   * @brief Construct a new Interpolator Base object.
   * 
   * No a-priori knowledge about number of levels in multigrid.
   * 
   */
  InterpolatorBase() {}

  /**
   * @brief Construct P and R matrices based on dofs and level information.
   * 
   * @param n_h_dofs Number of dofs in the finer level.
   * @param n_H_dofs Number of dofs in the coarser level.
   * @param level Current level.
   */
  virtual void make_operators(size_t n_h_dofs, size_t n_H_dofs,
                              size_t level) = 0;

  /**
   * @brief Prolongation operator on `v` and updating `result` inplace.
   * 
   * @param v 
   * @param level 
   */
  Eigen::Matrix<EleType, -1, 1> prolongation(
      const Eigen::Matrix<EleType, -1, 1>& v, size_t level) {
    Eigen::Matrix<EleType, -1, 1> result = get_P(level) * v;
    return result;
  }

  /**
   * @brief Restriction operator on `v`.
   * 
   * @param v 
   * @param level 
   */
  Eigen::Matrix<EleType, -1, 1> restriction(
      const Eigen::Matrix<EleType, -1, 1>& v, size_t level) {
    Eigen::Matrix<EleType, -1, 1> result = get_R(level) * v;
    return result;
  }

  const Eigen::SparseMatrix<EleType>& get_P(size_t level) const {
    return level_to_P[level];
  }

  const Eigen::SparseMatrix<EleType>& get_R(size_t level) const {
    return level_to_R[level];
  }

  void set_level_to_P(size_t level, Eigen::SparseMatrix<EleType>& P) {
    level_to_P[level] = P;
    return;
  }

  void set_level_to_R(size_t level, Eigen::SparseMatrix<EleType>& R) {
    level_to_R[level] = R;
    return;
  }
};

/**
 * @brief Interface for linear interpolation.
 * 
 * References:
 * 
 * [1] : Briggs2000. "Introduction to Algebraic Multigrid, 2ed.". Chapter 3.
 * 
 * @tparam EleType 
 */
template <class EleType>
class LinearInterpolator : public InterpolatorBase<EleType> {
 private:
  const size_t n_elements_per_columns = 3;

 public:
  using InterpolatorBase<EleType>::InterpolatorBase;

  void make_operators(size_t n_h_dofs, size_t n_H_dofs, size_t level) {
    // Create prolongation matrix
    size_t nnz = n_H_dofs * n_elements_per_columns;
    Eigen::SparseMatrix<EleType> P(n_h_dofs, n_H_dofs);
    P.reserve(nnz);

    // Populate nonzeros in matrix and bounds check the rows
    // TODO: Do these bound checks disrupt algorithm correctness?
    std::vector<Eigen::Triplet<EleType>> P_coefficients;
    P_coefficients.reserve(nnz);
    size_t i = 0;
    for (size_t j = 0; j < n_H_dofs; ++j) {
      if (i < n_h_dofs)
        P_coefficients.push_back(Eigen::Triplet<EleType>(i, j, 0.5));

      if (i + 1 < n_h_dofs)
        P_coefficients.push_back(Eigen::Triplet<EleType>(i + 1, j, 1.0));

      if (i + 2 < n_h_dofs)
        P_coefficients.push_back(Eigen::Triplet<EleType>(i + 2, j, 0.5));

      i += n_elements_per_columns - 1;
    }
    P.setFromTriplets(P_coefficients.begin(), P_coefficients.end());

    // Restriction matrix follows from prolongation matrix
    Eigen::SparseMatrix<EleType> R(n_H_dofs, n_h_dofs);
    R.reserve(P.nonZeros());
    R = P.transpose();

    // Update the maps
    this->set_level_to_P(level, P);
    this->set_level_to_R(level, R);

    return;
  }
};

/**
 * @brief Interface for direct interpolation using classical strength.
 * 
 * References:
 * 
 * [1] : PyAMG. url: https://github.com/pyamg/pyamg/blob/main/pyamg/strength.py
 * 
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