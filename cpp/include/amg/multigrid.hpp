#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <amg/common.hpp>
#include <amg/grid.hpp>
#include <amg/interpolator.hpp>
#include <amg/smoother.hpp>

namespace AMG {

template <class EleType>
class Multigrid {
 private:
  /**
   * @brief Interpolator providing restriction and prolongation operations.
   * 
   */
  InterpolatorBase<EleType>* interpolator;

  AMG::SmootherBase<EleType>* smoother;

  Eigen::SimplicialLDLT<Eigen::SparseMatrix<EleType>> coarse_direct_solver;

  /**
   * @brief Tolerance below which a smoother is considered to have converged.
   *
   */
  EleType tolerance;

  /**
   * @brief Compute the error every `n` iterations during smoothing.
   *
   */
  size_t compute_error_every_n_iters;

  /**
   * @brief Maximum number of iterations before smoothing termination.
   *
   */
  size_t n_iters;

  /**
   * @brief 
   * 
   */
  size_t n_fine_nodes;

  /**
   * @brief Number of multigrid levels where level 0 is finest.
   * 
   */
  size_t n_levels;

  /**
   * @brief Index of the finest grid.
   * 
   */
  size_t finest_grid_ix = 0;

  /**
   * @brief Index of the coarsest grid.
   * 
   */
  size_t coarsest_grid_ix;

  /**
   * @brief Map multigrid level to number of degrees of freedom for that level.
   * 
   * Note that `n_dofs==n_nodes*n_nodes`
   * 
   */
  std::vector<size_t> level_to_n_dofs;

  /**
   * @brief Map multigrid level to coefficient matrix A
   * 
   */
  std::vector<Eigen::SparseMatrix<EleType>> level_to_coefficient_matrix;

  /**
   * @brief Map multigrid level to right hand side (forcing) vector.
   * 
   */
  std::vector<Eigen::Matrix<EleType, -1, 1>> level_to_rhs;

  /**
   * @brief Map multigrid level to solution (u) vector.
   * 
   */
  std::vector<Eigen::Matrix<EleType, -1, 1>> level_to_soln;

  /**
   * @brief Map multigrid level to residual (e) vector.
   * 
   */
  std::vector<Eigen::Matrix<EleType, -1, 1>> level_to_residual;

  /**
   * @brief Return number of next coarsest dofs given finer dofs.
   * 
   * The next coarsest dofs formula follows from the following (see ref [1]):
   * 
   * ```
   * nc = n/2 - 1
   * nf = n - 1
   * # nc in terms of nf
   * nc = (nf + 1)/2 - 1
   * ```
   * 
   * References:
   * 
   * [1] : Briggs2000. "A Multigrid Tutorial, 2ed". pp. 34.
   * 
   * @return size_t 
   */
  size_t n_H_dofs_from_n_h_dofs(size_t h_dofs) {
    size_t H_dofs = (h_dofs + 1) / 2 - 1;
    return H_dofs;
  }

  bool display_error{false};

 public:
  ~Multigrid() {}

  Multigrid() = delete;

  /**
   * @brief Construct a new Multigrid object
   *
   * TODO: Provide finest grid n-nodes, n-levels, and maxiters convergence...
   * this information can be used by Grid objects to construct
   * the desired linear systems, right? Minimum number of levels also?? Or 
   *
   * @param smoother_
   * @param n_fine_nodes_ Used to compute gridspacing h for finest level.
   * @param n_levels_ Desired number of levels where level 0 is finest level.
   */
  Multigrid(AMG::InterpolatorBase<EleType>* interpolator_,
            AMG::SmootherBase<EleType>* smoother_, size_t n_fine_nodes_,
            size_t n_levels_, EleType tolerance_ = 1e-9,
            size_t compute_error_every_n_iters_ = 10, size_t n_iters_ = 100)
      : interpolator(interpolator_),
        smoother(smoother_),
        n_fine_nodes(n_fine_nodes_),
        n_levels(n_levels_),
        tolerance(tolerance_),
        compute_error_every_n_iters(compute_error_every_n_iters_),
        n_iters(n_iters_) {

    if (compute_error_every_n_iters > n_iters) {
      std::string msg =
          "`compute_error_every_n_iters` must be leq to `n_iters`, got " +
          std::to_string(compute_error_every_n_iters) + " and " +
          std::to_string(n_iters);
      throw(std::invalid_argument(msg));
    }

    // Initialize linear system info
    level_to_coefficient_matrix.resize(n_levels);
    level_to_soln.resize(n_levels);
    level_to_rhs.resize(n_levels);
    level_to_residual.resize(n_levels);
    level_to_n_dofs.resize(n_levels);

    // Initialize index for coarsest grid
    coarsest_grid_ix = n_levels - 1;

    // Initialize the finest coefficients matrix
    size_t n_fine_dofs = n_fine_nodes * n_fine_nodes;
    level_to_n_dofs[finest_grid_ix] = n_fine_dofs;
    level_to_coefficient_matrix[finest_grid_ix] =
        AMG::Grid<EleType>::laplacian(n_fine_nodes);

    // Initialize the finest solutions vector
    Eigen::Matrix<EleType, -1, 1> u(n_fine_dofs);
    u.setZero();
    level_to_soln[finest_grid_ix] = u;

    // Initialize the finest right hand side
    level_to_rhs[finest_grid_ix] = AMG::Grid<EleType>::rhs(n_fine_nodes);

    // Initialize the finest residual
    auto b = level_to_rhs[finest_grid_ix];
    auto A = level_to_coefficient_matrix[finest_grid_ix];
    level_to_residual[finest_grid_ix] = b - A * u;

    // Use the finest level grid info to construct the linear interpolators
    // needed to construct the remaining grid stuff
    // TODO: could do while coarsest is greater than max coarse ndofs
    size_t n_H_dofs;
    size_t n_h_dofs;
    for (size_t level = 1; level < n_levels; ++level) {
      // Make restriction/prolongation matrices
      n_h_dofs = level_to_n_dofs[level - 1];
      n_H_dofs = n_H_dofs_from_n_h_dofs(n_h_dofs);
      level_to_n_dofs[level] = n_H_dofs;
      interpolator->make_operators(n_h_dofs, n_H_dofs, level - 1);

      // Make coefficient matrix using interpolation operators
      auto R_h = interpolator->get_R(level - 1);
      auto A_h = level_to_coefficient_matrix[level - 1];  // finer matrix
      auto P_h = interpolator->get_P(level - 1);
      auto A_H = R_h * (A_h * P_h);  // coarser matrix
      level_to_coefficient_matrix[level] = A_H;

      // Make solution vector
      Eigen::Matrix<EleType, -1, 1> u_H(n_H_dofs);
      u_H.setZero();
      level_to_soln[level] = u_H;

      // Make right hand side
      Eigen::Matrix<EleType, -1, 1> rhs_H(n_H_dofs);
      rhs_H.setZero();
      level_to_rhs[level] = rhs_H;

      // Make residual vector
      level_to_residual[level] = rhs_H - A_H * u_H;
    }

    // Initialize the coarse solver
    coarse_direct_solver.analyzePattern(
        level_to_coefficient_matrix[coarsest_grid_ix]);
    coarse_direct_solver.factorize(
        level_to_coefficient_matrix[coarsest_grid_ix]);
  }

  /**
   * @brief A single multigrid vcycle.
   * 
   * Ref [1] was used for the main algorithm (non-recursive) while refs [2-4]
   * were used to verify that coarse solutions need to be zeroed at each 
   * vcycle.
   *
   * References:
   * 
   * [1] : [Algebraic Multigrid from amgcl](https://amgcl.readthedocs.io/en/latest/amg_overview.html)
   * [2] : Kostler2008. "Multigrid HowTo."
   * [3] : Pawar2019. "CFD Julia."
   * [4] : [AlgebraicMultigrid.jl solve](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/blob/84e4a6a2fb0e3d43ad64b1945f0d199feac356c4/src/multilevel.jl#L266)
   */
  void vcycle() {
    // At each level of the grid hiearchy, finest-to-coarsest:
    for (size_t level = 0; level < n_levels; ++level) {
      //  1. Apply a couple of smoothing iterations (pre-relaxation) to the
      // current solution ui=Si(Ai,fi,ui)
      smoother->smooth(level_to_coefficient_matrix[level], level_to_soln[level],
                       level_to_rhs[level]);

      //  2. Find residual ei=fi−Aiui and restrict it to the coarser level
      level_to_residual[level] =
          level_to_rhs[level] -
          level_to_coefficient_matrix[level] * level_to_soln[level];

      if (level + 1 != n_levels) {
        // initialize coarse solution to zero here refs [2-4].
        level_to_soln[level + 1].setZero();

        // f_{i+1} = R_{i} e_{i}
        level_to_rhs[level + 1] =
            interpolator->restriction(level_to_residual[level], level);
      }
    }

    // Solve the coarsest system directly: uL=A−1LfL
    level_to_soln[coarsest_grid_ix] =
        coarse_direct_solver.solve(level_to_rhs[coarsest_grid_ix]);

    // At each level of the grid hierarchy, coarsest-to-finest:
    for (int level = coarsest_grid_ix - 1; level >= 0; --level) {
      //  1. Update the current solution with the interpolated solution from the
      // coarser level: ui=ui+Piui+1
      level_to_soln[level] =
          level_to_soln[level] +
          interpolator->prolongation(level_to_soln[level + 1], level);

      //  2. Apply a couple of smoothing iterations (post-relaxation) to the
      // updated solution: ui=Si(Ai,fi,ui)
      smoother->smooth(level_to_coefficient_matrix[level], level_to_soln[level],
                       level_to_rhs[level]);
    }

    return;
  }

  /**
   * @brief Solve a linear system of equations via vcycle
   *
   */
  const Eigen::Matrix<EleType, -1, 1>& solve() {
    size_t iter = 0;
    EleType error = 100;
    auto A = get_coefficient_matrix(finest_grid_ix);
    auto b = get_rhs(finest_grid_ix);
    while (iter < n_iters && error > tolerance) {
      vcycle();
      iter += 1;
      if ((iter % compute_error_every_n_iters) == 0) {
        auto u = get_soln(finest_grid_ix);
        error = rss(A, u, b);

        if (display_error) {
          std::cout << "Iter: " << iter << " | Error: " << error << std::endl;
        }
      }
    }

    if (error <= tolerance)
      std::cout << "AMG converged after " << iter << " iterations."
                << std::endl;
    else
      std::cout << "AMG did not converge after " << iter << " iterations."
                << std::endl;

    return level_to_soln[finest_grid_ix];
  }

  const Eigen::SparseMatrix<EleType>& get_coefficient_matrix(
      size_t level) const {
    return level_to_coefficient_matrix[level];
  }

  const Eigen::Matrix<EleType, -1, 1>& get_soln(size_t level) const {
    return level_to_soln[level];
  }

  const Eigen::Matrix<EleType, -1, 1>& get_rhs(size_t level) const {
    return level_to_rhs[level];
  }

  const size_t get_n_dofs(size_t level) const { return level_to_n_dofs[level]; }

  const EleType get_tolerance() const { return tolerance; }

  void display_error_on() {
    display_error = true;
    return;
  }

  void display_error_off() {
    display_error = true;
    return;
  }
};

}  // end namespace AMG