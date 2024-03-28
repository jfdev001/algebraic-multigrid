#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <amg/common.hpp>

namespace AMG {

template <class EleType>
class SmootherBase {
 public:
  /**
     * @brief Tolerance below which a smoother is considered to have converged.
     *
     */
  EleType tolerance{1e-9};

  /**
     * @brief Compute the error every `n` iterations during smoothing.
     *
     */
  size_t compute_error_every_n_iters{100};

  /**
     * @brief Maximum number of iterations before smoothing termination.
     *
     */
  size_t n_iters{1000};

  SmootherBase() {}

  SmootherBase(size_t n_iters_) : n_iters(n_iters_) {}

  SmootherBase(double tolerance_, size_t compute_error_every_n_iters_,
               size_t n_iters_)
      : tolerance(tolerance_),
        compute_error_every_n_iters(compute_error_every_n_iters_),
        n_iters(n_iters_) {}

  /**
     * @brief Derived types must implement a `smooth` function that smooths `Au
     * = b`.
     *
     * @param A Coeffcients matrix for linear system of equations.
     * @param u Solution to linear system of equations.
     * @param b Right hand side of linear system of equations.
     */
  virtual void smooth(const Eigen::SparseMatrix<EleType>& A,
                      Eigen::Matrix<EleType, -1, 1>& u,
                      const Eigen::Matrix<EleType, -1, 1>& b) = 0;

  void set_n_iters(size_t n_iters_) { n_iters = n_iters_; }

  void set_tolerance(double tolerance_) { tolerance = tolerance_; }

  void set_compute_every_n_iters(size_t compute_error_every_n_iters_) {
    compute_error_every_n_iters = compute_error_every_n_iters_;
  }
};

template <class EleType>
class Jacobi : public SmootherBase<EleType> {
 public:
  // C++11
  // https://stackoverflow.com/questions/8093882/using-c-base-class-constructors
  using SmootherBase<EleType>::SmootherBase;

  Jacobi() {}

  /**
     * @brief Update initial guess `u` inplace using Jacobi method.
     *
     * References:
     *
     * [1] : Heath, M.T. Scientific Computing. pp 468. SIAM 2018.
     */
  void smooth(const Eigen::SparseMatrix<EleType>& A,
              Eigen::Matrix<EleType, -1, 1>& u,
              const Eigen::Matrix<EleType, -1, 1>& b) {
    size_t iter = 0;
    size_t ndofs = b.size();
    EleType error = 100;
    EleType sigma;
    while (iter < this->n_iters && error > this->tolerance) {
      for (size_t i = 0; i < ndofs; ++i) {
        sigma = 0;
        for (size_t j = 0; j < ndofs; ++j) {
          if (j != i) {
            sigma += A.coeff(i, j) * u[j];
          }
        }
        EleType aii = A.coeff(i, i);
        u[i] = (b[i] - sigma) / aii;
      }
      iter += 1;
      if (iter % this->compute_error_every_n_iters == 0) {
        error = rss(A, u, b);
      }
    }
    return;
  }
};

template <class EleType>
class SuccessiveOverRelaxation : public SmootherBase<EleType> {
 private:
  /**
     * @brief Relaxation parameter in [0, 2].
     *
     */
  double omega{1.0};

  /**
     * @brief Force `omega` to be in [0, 2].
     *
     * TODO: Is there a better way to handle this in the constructor?
     *
     */
  void validate_omega() {
    if (omega > 2 || omega < 0) {
      std::string msg =
          "`omega` must be in [0, 2] but got omega=" + std::to_string(omega) +
          "\n";
      throw std::invalid_argument(msg);
    }
  }

 public:
  // C++11
  // https://stackoverflow.com/questions/8093882/using-c-base-class-constructors
  using SmootherBase<EleType>::SmootherBase;

  SuccessiveOverRelaxation() {}
  /**
     * @brief Construct a new Successive Over Relaxation object
     *
     * This constructor only sets the `omega` member data and leaves the
     * Base class's member data alone.
     *
     * @param omega_
     */
  SuccessiveOverRelaxation(double omega_) : omega(omega_) { validate_omega(); }

  /**
     * @brief Construct a new Successive Over Relaxation object.
     *
     * This constructor also sets the Base class's member data.
     *
     * @param omega_
     * @param tolerance_
     * @param compute_error_every_n_iters_
     * @param n_iters_
     */
  SuccessiveOverRelaxation(double omega_, double tolerance_,
                           size_t compute_error_every_n_iters_, size_t n_iters_)
      : SmootherBase<EleType>(tolerance_, compute_error_every_n_iters_,
                              n_iters_),
        omega(omega_) {
    validate_omega();
  }

  /**
     * @brief Update initial guess `u` inplace with SOR and internal relaxation
     * param `omega`
     *
     * If `this.omega == 1`, then this is equivalent to a Gauss-Seidel smoother.
     *
     * References:
     *
     * [1] : Heath, M.T. Scientific Computing. pp 470. SIAM 2018.
     */
  void smooth(const Eigen::SparseMatrix<EleType>& A,
              Eigen::Matrix<EleType, -1, 1>& u,
              const Eigen::Matrix<EleType, -1, 1>& b) {
    size_t iter = 0;
    size_t ndofs = b.size();
    EleType error = 100;
    EleType sigma_j_less_i;
    EleType sigma_j_greater_i;
    while (iter < this->n_iters && error > this->tolerance) {
      for (size_t i = 0; i < ndofs; ++i) {
        sigma_j_less_i = 0;
        for (size_t j = 0; j < i; ++j) {
          sigma_j_less_i += A.coeff(i, j) * u[j];
        }

        sigma_j_greater_i = 0;
        for (size_t j = i + 1; j < ndofs; ++j) {
          sigma_j_greater_i += A.coeff(i, j) * u[j];
        }

        EleType aii = A.coeff(i, i);
        EleType uk_plus_one_gauss_seidel =
            (b[i] - sigma_j_less_i - sigma_j_greater_i) / aii;
        EleType uk = u[i];

        u[i] = uk + omega * (uk_plus_one_gauss_seidel - uk);
      }
      iter += 1;
      if (iter % this->compute_error_every_n_iters == 0) {
        error = rss(A, u, b);
      }
    }
    return;
  }
};

}  // end namespace AMG
