#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <iostream>

#include <amg/grid.hpp>
#include <amg/multigrid.hpp>
#include <amg/solver.hpp>
#include <amg/problem.hpp>

TEST_CASE("Quick check", "[main]") {
    unsigned int ndims = 2;
    Eigen::SparseMatrix<double> A = laplacian(ndims);
    std::cout << A << std::endl;
}