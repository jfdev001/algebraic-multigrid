#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <amg/common.hpp>

namespace AMG {

/**
 * @brief Base class for smoothers that must implement a `smooth` function
 * for the iterative solution of a linear system `Au = b`.
 * 
 * @tparam EleType 
 */
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
  size_t n_iters{1};

  SmootherBase() {}

  SmootherBase(size_t n_iters_) : n_iters(n_iters_) {}

  /**
   * @brief Construct a new Smoother Base object with iterative solver member data.
   * 
   * @param tolerance_ 
   * @param compute_error_every_n_iters_ 
   * @param n_iters_ 
   */
  SmootherBase(double tolerance_, size_t compute_error_every_n_iters_,
               size_t n_iters_)
      : tolerance(tolerance_),
        compute_error_every_n_iters(compute_error_every_n_iters_),
        n_iters(n_iters_) {}

  /**
   * @brief Must implement function that smooths `Au = b`.
   * 
   * @param A Coefficients matrix for discretized governing equations. Coeffcients matrix for linear system of equations.
   * @param u Solution to linear system of equations.
   * @param b Right hand side of linear system `Au = b`. Right hand side of linear system of equations.
   */
  virtual void smooth(const Eigen::SparseMatrix<EleType>& A,
                      Eigen::Matrix<EleType, -1, 1>& u,
                      const Eigen::Matrix<EleType, -1, 1>& b) = 0;
};

/**
 * @brief Symmetric Gauss-Seidel smoother for sparse systems.
 * 
 * The forward and backward sweep are not strictly both necessary, it seems like
 * most numerical linear algebra books (E.g., Heath's "Scientific Computing") 
 * only propose one sweep as the algorithm. For completeness, however, Ref [2] 
 * does assert the justification of the so-called symmetric sweep.
 * 
 * References
 * 
 * [1] : [smoother.jl in AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/blob/master/src/smoother.jl)
 * 
 * [2] : [Strang2006 MIT 18.086 Lecture: Iterative Methods](https://math.mit.edu/classes/18.086/2006/am62.pdf)
 * 
 * [3] : [Julia Lang: Dot product as linear combination](https://discourse.julialang.org/t/what-are-the-pros-and-cons-of-row-column-major-ordering/110045)
 * 
 * @tparam EleType 
 */
template <class EleType>
class SparseGaussSeidel : public SmootherBase<EleType> {
 private:
  /**
  * @brief Modified sparse matvec product of `Au` for Gauss-Seidel sweeps.
  * 
  * The results of this operation update `rsum` and `diag` inplace.
  * 
  * @param col Index of the column for which you desire sparse row indices and values.
  * @param rsum Sum representing the matvec product.
  * @param diag Value of diagonal for particular column of CSC matrix.
  * @param z Zero value
  * @param A Coefficients matrix for discretized governing equations. 
  * @param u Solution to linear system of equations. 
  */
  void matvecprod(int col, EleType& rsum, EleType& diag, const EleType& z,
                  const Eigen::SparseMatrix<EleType>& A,
                  const Eigen::Matrix<EleType, -1, 1>& u) {
    int row;
    EleType val;
    for (typename Eigen::SparseMatrix<EleType>::InnerIterator it(A, col); it;
         ++it) {
      row = it.row();
      val = it.value();
      diag = (col == row) // if you found a value on the diagonal, update it for other iters
                 ? val
                 : diag;  // column == row therefore val == A_ii on diag
      rsum += (col == row)
                  ? z
                  : val * u[row];  // contrib to sum should be 0 for i==j by def
    }
  }

  /**
   * @brief Updates a single entry in `u` inplace using the definition of a Gauss-Seidel.
   * 
   * \f$ u_{i} = \frac{b_i - \sum_{j \neq i} a_{ij} u_j }{a_{ii}}. \f$
   * 
   * @param col 
   * @param A Coefficients matrix for discretized governing equations. 
   * @param b Right hand side of linear system `Au = b`. 
   * @param u Solution to linear system of equations. 
   */
  void update(int col, const Eigen::SparseMatrix<EleType>& A,
              const Eigen::Matrix<EleType, -1, 1>& b,
              Eigen::Matrix<EleType, -1, 1>& u) {
    EleType z = 0;
    EleType rsum = z;
    EleType diag = z;
    matvecprod(col, rsum, diag, z, A, u);
    u[col] = diag == z ? u[col] : (b[col] - rsum) / diag;
    return;
  }

  /**
   * @brief Gauss-Seidel iteration starting from the first row.
   *    
   * @param A Coefficients matrix for discretized governing equations. 
   * @param b Right hand side of linear system `Au = b`. 
   * @param u Solution to linear system of equations. 
   * @param nrows 
   */
  void forwardsweep(const Eigen::SparseMatrix<EleType>& A,
                    const Eigen::Matrix<EleType, -1, 1>& b,
                    Eigen::Matrix<EleType, -1, 1>& u, const int& ncols) {

    // iterate through cols of A in forward direction
    for (int col = 0; col < ncols; ++col) {
      update(col, A, b, u);
    }
    return;
  }

  /**
   * @brief Gauss-Seidel iteration starting from the last row.
   * 
   * @param A Coefficients matrix for discretized governing equations. 
   * @param b Right hand side of linear system `Au = b`. 
   * @param u Solution to linear system of equations. 
   * @param nrows 
   */
  void backwardsweep(const Eigen::SparseMatrix<EleType>& A,
                     const Eigen::Matrix<EleType, -1, 1>& b,
                     Eigen::Matrix<EleType, -1, 1>& u, const int& ncols) {
    // iterate through cols A in the backward direction
    for (int col = ncols - 1; col >= 0; --col) {
      update(col, A, b, u);
    }
  }

 public:
  using SmootherBase<EleType>::SmootherBase;

  /**
   * @brief Construct a new Sparse Gauss Seidel object for pre/post smoother in AMG.
   * 
   */
  SparseGaussSeidel() {
    this->tolerance = 1e-9;
    this->compute_error_every_n_iters = 0;
    this->n_iters = 1;
  }

  void smooth(const Eigen::SparseMatrix<EleType>& A,
              Eigen::Matrix<EleType, -1, 1>& u,
              const Eigen::Matrix<EleType, -1, 1>& b) {
    int ncols = A.cols();
    size_t iter = 0;
    EleType error = 100;
    while (iter < this->n_iters && error > this->tolerance) {
      forwardsweep(A, b, u, ncols);
      backwardsweep(A, b, u, ncols);
      iter += 1;
      if (this->compute_error_every_n_iters != 0 &&
          iter % this->compute_error_every_n_iters == 0) {
        error = rss(A, u, b);
      }
    }

    if (this->compute_error_every_n_iters != 0) {
      if (error <= this->tolerance)
        std::cout << "SPGS converged after " << iter << " iterations."
                  << std::endl;
      else
        std::cout << "SPGS did not converge after " << iter << " iterations."
                  << std::endl;
    }

    return;
  }
};

/**
 * @brief Dense Jacobi iterative method.
 * 
 * @tparam EleType 
 */
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

/**
 * @brief Dense successive over relaxation iterative method.
 * 
 * @tparam EleType 
 */
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
