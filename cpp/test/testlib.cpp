#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <iostream>

#include <Eigen/SparseLU>

#include <amg/multigrid.hpp>
#include <amg/smoother.hpp>
#include <amg/problem.hpp>


TEST_CASE("All Tests", "[main]") {
    // Setup coefficients matrix
    size_t n_interior_points = 2;
    size_t ndofs = n_interior_points*n_interior_points;
    Eigen::SparseMatrix<double> A = laplacian(n_interior_points);

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
    Eigen::VectorXd b = rhs(domain_1D(Eigen::seq(1, Eigen::last-1)));
    REQUIRE(b.size() == ndofs); 

    // Use built-in solver for comparison solution
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> direct_solver;
    Eigen::VectorXd exact_u(ndofs); 
    direct_solver.analyzePattern(A);
    direct_solver.factorize(A);
    exact_u = direct_solver.solve(b);

    // Inspecting small problems
    if (ndofs <= 10) {
        std::cout << "-------\nA\n-------\n";
        std::cout << A << std::endl;
        std::cout << "-------\nu\n-------\n";
        std::cout << exact_u << std::endl;
        std::cout << "-------\nb\n-------\n";
        std::cout << b << std::endl;
    }

    // CHECK SOR smoother
    AMG::SuccessiveOverRelaxation<double> sor;

    // CHECK multigrid solver matches a builtin solver 
    AMG::Multigrid mg = AMG::Multigrid();
}
