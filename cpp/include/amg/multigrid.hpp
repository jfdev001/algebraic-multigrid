#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <amg/common.hpp>
#include <amg/grid.hpp>
#include <amg/smoother.hpp>

namespace AMG {

template <class EleType>
class Multigrid {
 private:
  /**
  * @brief Restrict the reisdual to the right hand side.
  * 
  * TODO: Copy?
  * 
  * @param e 
  * @return Eigen::Matrix<EleType, -1, 1> 
  */
  Eigen::Matrix<EleType, -1, 1> restriction(
      const Eigen::Matrix<EleType, -1, 1>& e) {
    return e;
  }

  /**
   * @brief Prolongate (interpolate) the solution from coarse to finer level.
   * 
   * @param u 
   * @return Eigen::Matrix<EleType, -1, 1> 
   */
  Eigen::Matrix<EleType, -1, 1> prolongation(
      const Eigen::Matrix<EleType, -1, 1>& u) {
    return u;
  }

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
   * @brief Grid level from finest to coarsest is h, 2h, 4h, ..., H
   * 
   */
  const size_t grid_spacing_factor = 2;

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
   * @brief Map multigrid level to the number of nodes in x or y direction.
   * 
   */
  std::vector<size_t> level_to_n_nodes;

  /**
   * @brief Map multigrid level to number of degrees of freedom for that level.
   * 
   * Note that `n_dofs==n_nodes*n_nodes`
   * 
   */
  std::vector<size_t> level_to_n_dofs;

  /**
   * @brief Map multigrid level to grid spacing h.
   * 
   * TODO: is this even used???
   * 
   */
  std::vector<EleType> level_to_grid_spacing;

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
  Multigrid(AMG::SmootherBase<EleType>* smoother_, size_t n_fine_nodes_,
            size_t n_levels_, EleType tolerance_ = 1e-9,
            size_t compute_error_every_n_iters_ = 100, size_t n_iters_ = 100)
      : smoother(smoother_),
        n_fine_nodes(n_fine_nodes_),
        n_levels(n_levels_),
        tolerance(tolerance_),
        compute_error_every_n_iters(compute_error_every_n_iters_),
        n_iters(n_iters_) {

    // Initialize grid info
    level_to_grid_spacing.resize(n_levels);
    level_to_n_nodes.resize(n_levels);
    level_to_n_dofs.resize(n_levels);

    // Initialize linear system info
    level_to_coefficient_matrix.resize(n_levels);
    level_to_soln.resize(n_levels);
    level_to_rhs.resize(n_levels);
    level_to_residual.resize(n_levels);

    // Initialize index for coarsest grid
    coarsest_grid_ix = n_levels - 1;

    // Initialize the finest level grid info in the multigrid
    level_to_grid_spacing[finest_grid_ix] =
        AMG::Grid<EleType>::grid_spacing_h(n_fine_nodes);
    level_to_n_nodes[finest_grid_ix] = n_fine_nodes;
    level_to_n_dofs[finest_grid_ix] = n_fine_nodes * n_fine_nodes;

    // fill the remaining coarse grid info
    for (size_t level = 1; level < n_levels; ++level) {
      EleType prev_level_grid_spacing = level_to_grid_spacing[level - 1];

      EleType cur_level_grid_spacing =
          grid_spacing_factor * prev_level_grid_spacing;

      size_t cur_level_n_nodes =
          AMG::Grid<EleType>::points_n_from_grid_spacing_h(
              cur_level_grid_spacing);

      level_to_grid_spacing[level] = cur_level_grid_spacing;
      level_to_n_nodes[level] = cur_level_n_nodes;
      level_to_n_dofs[level] = cur_level_n_nodes * cur_level_n_nodes;  // n**2
    }

    // Fill the linear system Au = b lists for different coarseness
    // TODO: are copy assignments being made?? efficiency here? w.r.t u and b init
    size_t n_nodes;
    size_t n_dofs;
    for (size_t level = 0; level < n_levels; ++level) {
      n_nodes = level_to_n_nodes[level];
      n_dofs = level_to_n_dofs[level];

      // Fill the fine through coarse coefficient array
      level_to_coefficient_matrix[level] =
          AMG::Grid<EleType>::laplacian(n_nodes);

      // Initialize the fine through coarse solution vector
      Eigen::Matrix<EleType, -1, 1> u(n_dofs);
      u.setZero();
      level_to_soln[level] = u;

      // Fill the fine through coarse right hand side array
      level_to_rhs[level] = AMG::Grid<EleType>::rhs(n_nodes);

      // Fill the residual
      auto b = level_to_rhs[level];
      auto A = level_to_coefficient_matrix[level];
      level_to_residual[level] = b - A * u;
    }

    // Initialize the coarse solver
    coarse_direct_solver.analyzePattern(
        level_to_coefficient_matrix[coarsest_grid_ix]);
    coarse_direct_solver.factorize(
        level_to_coefficient_matrix[coarsest_grid_ix]);
  }

  /**
   * @brief A single multigrid cycle.
   * 
   * TODO: really slow
   *
   * References:
   * 
   * [1] : [Algebraic Multigrid from amgcl](https://amgcl.readthedocs.io/en/latest/amg_overview.html)
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
        //level_to_rhs[level + 1] = restriction(level_to_residual[level]);
      }
    }

    // Solve the coarsest system directly: uL=A−1LfL
    level_to_soln[coarsest_grid_ix] =
        coarse_direct_solver.solve(level_to_rhs[coarsest_grid_ix]);

    // At each level of the grid hiearchy, coarsest-to-finest:
    for (long level = coarsest_grid_ix - 1; level >= 0; --level) {
      //  1. Update the current solution with the interpolated solution from the
      // coarser level: ui=ui+Piui+1
      //level_to_soln[level] =
      //    level_to_soln[level] + prolongation(level_to_soln[level + 1])

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
   * TODO: There should be an upper limit on the number of cycles/niters and/or tolerance
   * 
   */
  const Eigen::Matrix<EleType, -1, 1>& solve() {
    size_t iter = 0;
    EleType error = 100;
    while (iter < n_iters && error > tolerance) {
      std::cout << "iter: " << iter << std::endl;
      vcycle();
      iter += 1;
      if (iter % compute_error_every_n_iters == 0) {
        //error = residual(A, u, b);
      }
    }
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

  const EleType get_grid_spacing(size_t level) const {
    return level_to_grid_spacing[level];
  }

  const size_t get_n_nodes(size_t level) const {
    return level_to_n_nodes[level];
  }

  const size_t get_n_dofs(size_t level) const { return level_to_n_dofs[level]; }

  const EleType get_tolerance() const { return tolerance; }
};

}  // end namespace AMG