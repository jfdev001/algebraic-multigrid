#ifndef SMOOTHER_HPP
#define SMOOTHER_HPP

#include <Eigen/Core>
#include <Eigen/Sparse>

namespace AMG {

template<class EleType>
class SmootherBase
{
public:
    /**
     * @brief Derived types must implement a `smooth` function that smooths `Au = b`.
     * 
     * Pure virtual function.
     */
    virtual void smooth (
        const Eigen::SparseMatrix<EleType>& A, 
        Eigen::Matrix<EleType, -1, 1>& u,
        const Eigen::Matrix<EleType, -1, 1>& b,
        const size_t niters
    ) = 0;
};

template <class EleType>
class Jacobi : public SmootherBase<EleType>
{
public:
    Jacobi() {};
    void smooth (
        const Eigen::SparseMatrix<EleType>& A,  
        Eigen::Matrix<EleType, -1, 1>& u,
        const Eigen::Matrix<EleType, -1, 1>& b, 
        const size_t niters              
    );
};

template <class EleType>
class SuccessiveOverRelaxation : public SmootherBase<EleType>
{
private:
    double omega {1.0};
public:
    SuccessiveOverRelaxation();
    SuccessiveOverRelaxation(double omega_);
    void smooth (
        const Eigen::SparseMatrix<EleType>& A, 
        Eigen::Matrix<EleType, -1, 1>& u,
        const Eigen::Matrix<EleType, -1, 1>& b,
        const size_t niters
    ) override;
};

} // end namespace AMG
#endif
