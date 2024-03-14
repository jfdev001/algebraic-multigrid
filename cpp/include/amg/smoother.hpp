#pragma once

#include <string>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <amg/common.hpp>

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
    /**
     * @brief Update initial guess `u` inplace using Jacobi method.
     * 
     * @param A 
     * @param u 
     * @param b 
     * @param niters 
     */
    void smooth (
        const Eigen::SparseMatrix<EleType>& A,  
        Eigen::Matrix<EleType, -1, 1>& u,
        const Eigen::Matrix<EleType, -1, 1>& b, 
        const size_t niters              
    ) {
        return;
    }
};

template <class EleType>
class SuccessiveOverRelaxation : public SmootherBase<EleType>
{
private:
    double omega {1.0};
public:
    SuccessiveOverRelaxation() { }
    SuccessiveOverRelaxation(double omega_) : omega(omega_) { 
        if (omega > 2 || omega < 0) {
            std::string msg = "`omega` must be in [0, 2] but got omega=" + 
                std::to_string(omega) + "\n";
            throw std::invalid_argument(msg);
        }
    }

    /**
     * @brief Update initial guess `u` inplace with SOR and internal relaxation param `omega`
     * 
     * If `this.omega == 1`, then this is equivalent to a Gauss-Seidel smoother.
     * 
     * @param A 
     * @param u 
     * @param b 
     * @param niters 
     */
    void smooth (
        const Eigen::SparseMatrix<EleType>& A, 
        Eigen::Matrix<EleType, -1, 1>& u,
        const Eigen::Matrix<EleType, -1, 1>& b,
        const size_t niters
    ) {
        return;
    }
};

} // end namespace AMG
