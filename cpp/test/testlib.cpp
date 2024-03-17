#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <iostream>

#include <Eigen/Core>
#include <Eigen/SparseLU>

#include <amg/multigrid.hpp>
#include <amg/smoother.hpp>
#include <amg/problem.hpp>

// TODO: break up tests into smoother tests and multigrid tests
TEST_CASE("All Tests", "[main]") {
    // Setup coefficients matrix
    size_t n_interior_points = 2;
    size_t ndofs = n_interior_points*n_interior_points;
    Eigen::SparseMatrix<double> A = AMG::laplacian(n_interior_points);

    // Setup right hand side 
    size_t n_boundary_points = 2;
    size_t n_points_in_direction = n_interior_points + n_boundary_points;
    double left_bound = -1.0;
    double right_bound = 1.0;
    Eigen::VectorXd domain_1D = Eigen::VectorXd::LinSpaced(
        n_points_in_direction,
        left_bound,
        right_bound
    );
    Eigen::VectorXd b = AMG::rhs(domain_1D(Eigen::seq(1, Eigen::last-1)));
    REQUIRE(b.size() == ndofs); 

    // Use built-in solver for comparison solution
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> direct_solver;
    Eigen::VectorXd exact_u(ndofs); 
    direct_solver.analyzePattern(A);
    direct_solver.factorize(A);
    exact_u = direct_solver.solve(b);

    // Compute the residual
    double residual = AMG::residual(A, exact_u, b);
    std::cout << "Residual: " << residual << std::endl;

    // Inspecting small problems
    if (ndofs <= 10) {
        std::cout << "BEGIN Exact Solution:\n";
        std::cout << "-------\nA\n-------\n";
        std::cout << A << std::endl;
        std::cout << "-------\nu\n-------\n";
        std::cout << exact_u << std::endl;
        std::cout << "-------\nb\n-------\n";
        std::cout << b << std::endl;
        std::cout << "END Exact Solution\n";
    }

    // Valid SOR instantiation
    double bad_omega_less_than_0 = -0.01;
    double bad_omega_greater_than_2 = 2.01;
    using bad_sor = AMG::SuccessiveOverRelaxation<double>;

    CHECK_THROWS_AS(
        bad_sor(bad_omega_less_than_0), 
        std::invalid_argument
    );

    CHECK_THROWS_AS(
        bad_sor(bad_omega_greater_than_2),
        std::invalid_argument
    );

    // Check Jacobi smoother matches exact solution
    Eigen::VectorXd jacobi_u(ndofs); 
    jacobi_u.setZero();
    AMG::Jacobi<double> jacobi;
    jacobi.smooth(A, jacobi_u, b);
    CHECK(jacobi_u.isApprox(exact_u, jacobi.tolerance));

    if (ndofs < 10) {
        std::cout << "BEGIN jacobi solution:\n";
        std::cout << jacobi_u << std::endl;
        std::cout << "END jacobi solution\n";
    }

    // Check SOR smoother matches exact solution
    Eigen::VectorXd sor_u(ndofs); 
    sor_u.setZero();
    AMG::SuccessiveOverRelaxation<double> sor;
    sor.smooth(A, sor_u, b);
    CHECK(sor_u.isApprox(exact_u, sor.tolerance));

    if (ndofs < 10) {
        std::cout << "BEGIN SOR solution:\n";
        std::cout << sor_u << std::endl;
        std::cout << "END SOR solution\n";
    }

     // Instantiate sor/jacobi smoother using Base constructor
    double tolerance = 1e-10;
    size_t compute_error_every_n_iters = 100;
    size_t niters = 100;
    AMG::Jacobi<double> jacobi_base(tolerance, compute_error_every_n_iters, niters);
    AMG::SuccessiveOverRelaxation<double> sor_base(
        tolerance, compute_error_every_n_iters, niters);

    // // Valid multigrid instantiation
    // AMG::SuccessiveOverRelaxation<double> sor;
    // AMG::Multigrid<double> mg(&sor); // ref to derived class satisfies abstract arg req

    // // CHECK multigrid solver matches a builtin solver 
    // AMG::Multigrid mg = AMG::Multigrid();
}
