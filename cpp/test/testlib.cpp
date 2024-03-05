#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <iostream>

#include <amg/grid.hpp>
#include <amg/multigrid.hpp>
#include <amg/smoother.hpp>
#include <amg/problem.hpp>

TEST_CASE("All Tests", "[main]") {
    // Setup example problem
    unsigned int n_points_in_direction = 4;
    unsigned int ndofs = n_points_in_direction*n_points_in_direction;
    Eigen::SparseMatrix<double> A = laplacian(n_points_in_direction);
    Eigen::VectorXd b = rhs(ndofs);

    // CHECK gauss smoother

    // CHECK multigrid solver matches a builtin solver 
}
