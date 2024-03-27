#pragma once

#include <memory>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <amg/common.hpp>
#include <amg/grid.hpp>
#include <amg/smoother.hpp>

namespace AMG {

template <class EleType>
class Multigrid {
 private:
  void prolongation() { return; }  // to implement

  void restriction() { return; }  // to implement

  AMG::SmootherBase<EleType>* smoother;

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
  std::unique_ptr<size_t[]> level_to_n_nodes;

  /**
   * @brief Map multigrid level to number of degrees of freedom for that level.
   * 
   * Note that `n_dofs==n_nodes*n_nodes`
   * 
   */
  std::unique_ptr<size_t[]> level_to_n_dofs;

  /**
   * @brief Map multigrid level to grid spacing h.
   * 
   * TODO: is this even used???
   * 
   */
  std::unique_ptr<EleType[]> level_to_grid_spacing;

  /**
   * @brief Map multigrid level to coefficient matrix A
   * 
   */
  std::unique_ptr<Eigen::SparseMatrix<EleType>[]> level_to_coefficient_matrix;

  /**
   * @brief Map multigrid level to right hand side (forcing) vector.
   * 
   */
  std::unique_ptr<Eigen::Matrix<EleType, -1, 1>[]> level_to_rhs;

  /**
   * @brief Map multigrid level to solution (u) vector.
   * 
   */
  std::unique_ptr<Eigen::Matrix<EleType, -1, 1>[]> level_to_soln;

 public:
  ~Multigrid() {}

  Multigrid() = delete;

  /**
   * @brief Construct a new Multigrid object
   *
   * TODO: Provide finest grid n-nodes, n-levels, and maxiters convergence...
   * this information can be used by Grid objects to construct
   * the desired linear systems, right?
   *
   * @param smoother_
   * @param n_fine_nodes_ Used to compute gridspacing h for finest level.
   * @param n_levels_ Desired number of levels where level 0 is finest level.
   */
  Multigrid(AMG::SmootherBase<EleType>* smoother_, size_t n_fine_nodes_,
            size_t n_levels_)
      : smoother(smoother_), n_fine_nodes(n_fine_nodes_), n_levels(n_levels_) {

    // Initialize grid info
    level_to_grid_spacing = std::make_unique<EleType[]>(n_levels);
    level_to_n_nodes = std::make_unique<size_t[]>(n_levels);
    level_to_n_dofs = std::make_unique<size_t[]>(n_levels);

    // Initialize linear system info
    level_to_coefficient_matrix =
        std::make_unique<Eigen::SparseMatrix<EleType>[]>(n_levels);
    level_to_soln = std::make_unique<Eigen::Matrix<EleType, -1, 1>[]>(n_levels);
    level_to_rhs = std::make_unique<Eigen::Matrix<EleType, -1, 1>[]>(n_levels);

    // Initialize index for coarsest grid
    coarsest_grid_ix = n_levels - 1;

    // Initialize the finest level grid info in the multigrid
    level_to_grid_spacing[finest_grid_ix] =
        AMG::Grid<EleType>::grid_spacing_h(n_fine_nodes);
    level_to_n_nodes[finest_grid_ix] = n_fine_nodes;

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
    }
  }

  /**
   * @brief A single multigrid cycle.
   *
   */
  void vcycle() { return; }  // to implement

  /**
   * @brief Solve a linear system of equations via vcycle
   *
   * TODO: There should be an upper limit on the number of cycles/niters and/or tolerance
   * 
   */
  void solve() { return; }
};

}  // end namespace AMG