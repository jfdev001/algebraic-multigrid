// Example problem
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <unsupported/Eigen/KroneckerProduct>

/**
 * @brief Return second order central difference as linear operator on 1D function.
 * 
 * @param n Number of grid points in the x or y direction.
 * @return Eigen::SparseMatrix<double>
 */
Eigen::SparseMatrix<double> second_order_central_difference(size_t n) {
    // Grid spacing
    double h = 2.0 / (n + 1);

    // Unfilled difference matrix
    Eigen::SparseMatrix<double> D(n, n);    

    // Assemble the difference matrix by direct insertion of values
    size_t n_diagonals = 3;
    D.reserve(Eigen::VectorXi::Constant(n, n_diagonals));
    for (int i = 0; i < n; ++i) {
        D.insert(i, i) = -2.0;
        if (i > 0) { // handle first row
            D.insert(i, i - 1) = 1.0;
        }
        if (i < n - 1) { // handle last row
            D.insert(i, i + 1) = 1.0;
        }
    }
    
    // Finite differences requires this division
    D = D / (h * h);
    
    return D;
}

/**
 * @brief Return laplacian as linear operator on u(x,y) assuming homogenous dirichlet BCs.
 * 
 * References:
 *
 * [1] : [MIT Intro Linear PDEs](https://github.com/mitmath/18303/blob/master/supp_material/poissonFD.ipynb)
 * 
 * @param n Number of grid points in the x or y direction.
 * @return Eigen::SparseMatrix<double> 
 */
Eigen::SparseMatrix<double> laplacian(size_t n) {
    Eigen::SparseMatrix<double> D = second_order_central_difference(n);

    Eigen::SparseMatrix<double> spidentity(n, n);
    spidentity.setIdentity();

    Eigen::SparseMatrix<double> A = Eigen::kroneckerProduct(spidentity, D) + 
        Eigen::kroneckerProduct(D, spidentity);

    return A;
}

/**
 * @brief 
 * 
 * right hand side (see julia pde for iterating over the appropriate grid) 
 * 
 * @param ndofs 
 * @return Eigen::VectorXd 
 */
Eigen::VectorXd rhs(size_t ndofs) {
    return;
}
