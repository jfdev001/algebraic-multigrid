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
  size_t finest_grid = 0;

  /**
   * @brief Index of the coarsest grid.
   * 
   */
  size_t coarsest_grid;

  /**
   * @brief Map multigrid level to the number of nodes in x or y direction.
   * 
   */
  std::unique_ptr<size_t[]> level_to_n_nodes;

  /**
   * @brief Map multigrid level to grid spacing h.
   * 
   */
  std::unique_ptr<EleType[]> level_to_grid_spacing;

  /**
   * @brief Map multigrid level to coefficient matrix
   * 
   */
  std::unique_ptr<Eigen::SparseMatrix<EleType>[]> level_to_A;

  /**
   * @brief 
   * 
   * TODO:
   * 
   */
  //std::unique_ptr<generic vector[]>  level_to_rhs;

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

    level_to_grid_spacing = std::make_unique<EleType[]>(n_levels);
    level_to_n_nodes = std::make_unique<size_t[]>(n_levels);

    coarsest_grid = n_levels - 1;

    // initialize the finest level info
    level_to_grid_spacing[0] = AMG::Grid<EleType>::grid_spacing_h(n_fine_nodes);
    level_to_n_nodes[0] = n_fine_nodes;

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
    }
  }

  /**
     * @brief A single multigrid cycle.
     *
     */
  void vcycle() { return; }  // to implement

  /**
     * @brief Update `u` inplace but sucessively performing multigrid `vcycle`s.
     *
     * @param A
     * @param u
     * @param b
     * @param niters
     */
  void solve(const Eigen::SparseMatrix<EleType>& A,
             Eigen::Matrix<EleType, -1, 1>& u,
             const Eigen::Matrix<EleType, -1, 1>& b, const size_t niters) {
    return;
  }
};

}  // end namespace AMG