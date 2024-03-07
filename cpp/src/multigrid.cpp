#include <amg/multigrid.hpp>
#include <amg/smoother.hpp>

template <class EleType>
AMG::Multigrid<EleType>::Multigrid() {}

template <class EleType>
AMG::Multigrid<EleType>::Multigrid(
    AMG::SmootherBase<EleType>* smoother_
) : smoother(smoother_) {} 

template<class EleType>
void AMG::Multigrid<EleType>::solve (
    const Eigen::SparseMatrix<EleType>& A,
    Eigen::Matrix<EleType, -1, 1>& u,
    const Eigen::Matrix<EleType, -1, 1>& b,
    const size_t niters
) {
    //todo
    return;
}
