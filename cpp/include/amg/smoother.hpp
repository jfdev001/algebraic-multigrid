#include <Eigen/Core>
#include <Eigen/Sparse>

namespace AMG {

// todo: make base class requiring overloads of inherited jacobi, gauss-seidel, sor methods
class Smoother
{
private:
public:
    Smoother();
    ~Smoother();

    template<class T>
    Eigen::Matrix<T, -1, 1> smooth (
        const Eigen::SparseMatrix<T>& A, 
        const Eigen::Matrix<T, -1, 1>& b,
        const size_t niters
        //variable args here??
    );
};

} // end namespace AMG
