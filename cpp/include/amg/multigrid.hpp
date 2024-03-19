#pragma once

#include <memory>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <amg/smoother.hpp>
#include <amg/grid.hpp>
#include <amg/common.hpp>

namespace AMG {

template <class EleType>
class Multigrid {
private:
    void prolongation() { return; } //to implement
    
    void restriction() { return;} // to implement

    AMG::SmootherBase<EleType>* smoother;

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
     */
    Multigrid(AMG::SmootherBase<EleType>* smoother_) : smoother(smoother_){
        size_t n_fine_nodes; // use this to compute h
        size_t n_levels;     // should be used to compute coarse grid h's (2h, 4h, .. H)
        std::unique_ptr<EleType[]> grid_spacings(new EleType[n_levels]);
    }

    /**
     * @brief A single multigrid cycle.
     * 
     */
    void vcycle() {return; } // to implement

    /**
     * @brief Update `u` inplace but sucessively performing multigrid `vcycle`s.
     * 
     * @param A 
     * @param u 
     * @param b 
     * @param niters 
     */
    void solve (
        const Eigen::SparseMatrix<EleType>& A,
        Eigen::Matrix<EleType, -1, 1>& u,
        const Eigen::Matrix<EleType, -1, 1>& b,
        const size_t niters
    ) {
        return;
    }
};

} // end namespace AMG