#include <amg/smoother.hpp>

template <class EleType>
void AMG::Jacobi<EleType>::smooth (
        const Eigen::SparseMatrix<EleType>& A,  
        Eigen::Matrix<EleType, -1, 1>& u,
        const Eigen::Matrix<EleType, -1, 1>& b, 
        const size_t niters         
) {
    return;
}

template <class EleType>
AMG::SuccessiveOverRelaxation<EleType>::SuccessiveOverRelaxation() {}

template <class EleType>
AMG::SuccessiveOverRelaxation<EleType>::SuccessiveOverRelaxation(
    double omega_) : omega(omega_) {
    if (omega > 2 || omega < 0) {
        std::stringstream msg;
        msg << "`omega` must be in [0, 2] but got omega=" << omega << std::endl;
        throw std::invalid_argument(msg);
    }
}

template <class EleType>
void AMG::SuccessiveOverRelaxation<EleType>::smooth (
    const Eigen::SparseMatrix<EleType>& A, 
    Eigen::Matrix<EleType, -1, 1>& u,
    const Eigen::Matrix<EleType, -1, 1>& b,
    const size_t niters
) {
    return;
}
