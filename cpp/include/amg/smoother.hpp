#include <Eigen/Core>
#include <Eigen/Sparse>

namespace AMG {

class SmootherBase
{
public:
    /**
     * @brief Derived types must implement a `smooth` function that smoothes `Au = b`.
     * 
     */
    virtual Eigen::Matrix<double, -1, 1> smooth (
        const Eigen::SparseMatrix<double>& A, 
        const Eigen::Matrix<double, -1, 1>& u0,
        const Eigen::Matrix<double, -1, 1>& b,
        const size_t niters
    ) = 0;
};

class SuccessiveOverRelaxation : public SmootherBase
{
private:
public:
    Eigen::Matrix<double, -1, 1> smooth (
        const Eigen::SparseMatrix<double>& A, 
        const Eigen::Matrix<double, -1, 1>& u0,
        const Eigen::Matrix<double, -1, 1>& b,
        const size_t niters,
        const float omega
    );
};

} // end namespace AMG
