#include <amg/smoother.hpp>

// TODO: better way??
Eigen::Matrix<double, -1, 1> AMG::SuccessiveOverRelaxation::smooth (
    const Eigen::SparseMatrix<double>& A, 
    const Eigen::Matrix<double, -1, 1>& u0,
    const Eigen::Matrix<double, -1, 1>& b,
    const size_t niters,
    const float omega
) {
    // todo
}
