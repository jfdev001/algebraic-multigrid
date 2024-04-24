// Example problem: poisson equation w/ dirichlet boundary conditions
#pragma once

#include <cmath>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <unsupported/Eigen/KroneckerProduct>

namespace AMG {

/**
 * @brief Static functions for components of linear system `Au = b`.
 * 
 * @tparam EleType 
 */
template <class EleType>
class Grid {
 private:
  static const size_t n_boundary_points = 2;

 public:
  /**
   * @brief Compute the gridspacing h of a domain given n points in a direction.
   *
   * @param n
   * @return EleType
   */
  static EleType grid_spacing_h(size_t n) { return 2.0 / (n + 1); }

  /**
   * @brief Compute the number of points in a direction given gridspacing h.
   *
   * @param h
   * @return size_t
   */
  static size_t points_n_from_grid_spacing_h(EleType h = 1. / 50) {
    return static_cast<size_t>((2 / h) - 1);
  }

  /**
   * @brief Return second order central difference as linear operator on 1D
   * function.
   *
   * @param n Number of grid points in the x or y direction.
   * @return Eigen::SparseMatrix<EleType>
   */
  static Eigen::SparseMatrix<EleType> second_order_central_difference(
      size_t n) {
    // Grid spacing
    EleType h = grid_spacing_h(n);

    // Unfilled difference matrix
    Eigen::SparseMatrix<EleType> D(n, n);

    // Assemble the difference matrix by direct insertion of values
    size_t n_diagonals = 3;
    D.reserve(Eigen::VectorXi::Constant(n, n_diagonals));
    for (int i = 0; i < n; ++i) {
      D.insert(i, i) = -2.0;
      if (i > 0) {  // handle first row
        D.insert(i, i - 1) = 1.0;
      }
      if (i < n - 1) {  // handle last row
        D.insert(i, i + 1) = 1.0;
      }
    }

    // Finite differences requires this division
    D = D / (h * h);

    return D;
  }

  /**
   * @brief Return coefficients matrix `A` for laplacian as linear operator on 
   * u(x,y) assuming homogenous dirichlet BCs.
   *
   * References:
   *
   * [1] : [MIT Intro Linear PDEs](https://github.com/mitmath/18303/blob/master/supp_material/poissonFD.ipynb)
   *
   * @param n Number of grid points in the x or y direction.
   * @return Eigen::SparseMatrix<EleType>
   */
  static Eigen::SparseMatrix<EleType> laplacian(size_t n) {
    Eigen::SparseMatrix<EleType> D = second_order_central_difference(n);

    Eigen::SparseMatrix<EleType> spidentity(n, n);
    spidentity.setIdentity();

    Eigen::SparseMatrix<EleType> A = Eigen::kroneckerProduct(spidentity, D) +
                                     Eigen::kroneckerProduct(D, spidentity);

    return A;
  }

  /**
   * @brief Return right hand side vector `b` by evaluating `f` on a mesh grid 
   * in [-1, 1]^2.
   *    
   * @param n Number of interior grid points in the x or y direction.
   * @param f Function to be evaluated at each mesh grid point.
   * @return Eigen::Matrix<EleType, -1, 1>
   */
  static Eigen::Matrix<EleType, -1, 1> rhs(
      size_t n,
      std::function<EleType(EleType, EleType)> f = [](EleType x, EleType y) {
        return 5 * exp(-10 * (x * x + y * y));
      }) {

    // Initialize default 1D domain on [-1, 1]
    size_t n_points_in_direction = n + n_boundary_points;
    EleType left_bound = -1.0;
    EleType right_bound = 1.0;
    Eigen::Matrix<EleType, -1, 1> domain_1D =
        Eigen::DenseBase<Eigen::Matrix<EleType, -1, 1>>::LinSpaced(
            n_points_in_direction, left_bound, right_bound);

    // Initialize the rhs vector using the size of the 1D domain
    size_t ndofs = n * n;
    Eigen::Matrix<EleType, -1, 1> b(ndofs);

    // Evaluate the function at each interior grid point, traversing grid as col major
    size_t dof = 0;
    EleType xj, xi, feval;
    for (size_t j = 1; j <= n; ++j) {
      xj = domain_1D[j];
      for (size_t i = 1; i <= n; ++i) {
        xi = domain_1D[i];
        feval = f(xj, xi);
        b[dof] = feval;
        ++dof;
      }
    }

    return b;
  }
};

}  // end namespace AMG