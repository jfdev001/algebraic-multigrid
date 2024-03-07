#ifndef MULTIGRID_HPP
#define MULTIGRID_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <amg/smoother.hpp>

namespace AMG {

template <class EleType>
class Multigrid {
    private:
        template<class T>
        T prolongation(); //to implement
        
        template<class T>
        T restriction(); // to implement

        AMG::SmootherBase<EleType>* smoother;

    public:
        Multigrid();
        Multigrid(AMG::SmootherBase<EleType>* smoother_);

        template<class T>
        T vcycle(); // to implement

        void solve (
            const Eigen::SparseMatrix<EleType>& A,
            Eigen::Matrix<EleType, -1, 1>& u,
            const Eigen::Matrix<EleType, -1, 1>& b,
            const size_t niters
        );
};

} // end namespace AMG
#endif 