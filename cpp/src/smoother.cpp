#include <amg/smoother.hpp>

AMG::Smoother::Smoother() { /*todo*/ }

AMG::Smoother::~Smoother() {/*todo*/}

template<class T>
Eigen::Matrix<T, -1, 1> smooth (
    const Eigen::SparseMatrix<T>& A, 
    const Eigen::Matrix<T, -1, 1>& b,
    const size_t niters
) {
    // todo
}
