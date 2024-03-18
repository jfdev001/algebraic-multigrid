#pragma once

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
    Multigrid(AMG::SmootherBase<EleType>* smoother_) : smoother(smoother_){}

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