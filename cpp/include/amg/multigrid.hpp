#pragma once

#include <array>

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
  size_t n_fine_nodes;
  size_t n_levels;
  std::array<EleType, n_levels> grid_spacings;

 public:
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
      : smoother(smoother_), n_fine_nodes(n_fine_nodes_), n_levels(n_levels) {
    if (n_levels < 0) {
      std::string msg = "`n_levels` must be 0 or greater but got n_levels=" +
                        std::to_string(n_levels) + "\n";
      throw std::invalid_argument(msg);
    }
    grid_spacings // figure this shit out
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