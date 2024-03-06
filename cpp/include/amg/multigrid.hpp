#include <Eigen/Core>
#include <Eigen/Sparse>

namespace AMG {

class Multigrid {
    private:
        template<class T>
        T prolongation(); //to implement
        
        template<class T>
        T restriction(); // to implement

    public:
        Multigrid();
        ~Multigrid();

        template<class T>
        T vcycle(); // to imeplement

        template<class T>
        Eigen::Matrix<T, -1, 1> solve (
            const Eigen::SparseMatrix<T>& A, const Eigen::Matrix<T, -1, 1>& b
        );
};

} // end namespace AMG