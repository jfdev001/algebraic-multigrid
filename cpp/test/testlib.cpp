#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <amg/grid.hpp>
#include <amg/multigrid.hpp>
#include <amg/smoother.hpp>

// TODO: break up tests into smoother tests and multigrid tests
TEST_CASE("All Tests", "[main]") {
  // Setup coefficients matrix
  size_t n_interior_points = 2;
  Eigen::SparseMatrix<double> A =
      AMG::Grid<double>::laplacian(n_interior_points);

  // Setup right hand side
  Eigen::VectorXd b = AMG::Grid<double>::rhs(n_interior_points);

  // Verify dimension of rhs
  size_t ndofs = n_interior_points * n_interior_points;
  REQUIRE(b.size() == ndofs);

  // Use built-in solver for comparison solution
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> direct_solver;
  Eigen::VectorXd exact_u(ndofs);
  direct_solver.analyzePattern(A);
  direct_solver.factorize(A);
  exact_u = direct_solver.solve(b);

  // Compute the residual sum of squares
  double rss = AMG::rss(A, exact_u, b);
  std::cout << "RSS: " << rss << std::endl;

  // compute the residual
  Eigen::VectorXd residual(ndofs);
  residual = b - A * exact_u;

  // Inspecting small problems
  if (ndofs <= 10) {
    std::cout << "BEGIN Exact Solution:\n";
    std::cout << "-------\nA\n-------\n";
    std::cout << A << std::endl;
    std::cout << "-------\nu\n-------\n";
    std::cout << exact_u << std::endl;
    std::cout << "-------\nb\n-------\n";
    std::cout << b << std::endl;
    std::cout << "-------\ne\n-------\n";
    std::cout << residual << std::endl;
    std::cout << "END Exact Solution\n";
  }

  // Inspecting casting and grid sizes
  double h_from_n = AMG::Grid<double>::grid_spacing_h(n_interior_points);
  size_t n_from_h = AMG::Grid<double>::points_n_from_grid_spacing_h(h_from_n);
  CHECK(n_from_h == n_interior_points);

  // Valid SOR instantiation
  double bad_omega_less_than_0 = -0.01;
  double bad_omega_greater_than_2 = 2.01;
  using bad_sor = AMG::SuccessiveOverRelaxation<double>;

  CHECK_THROWS_AS(bad_sor(bad_omega_less_than_0), std::invalid_argument);

  CHECK_THROWS_AS(bad_sor(bad_omega_greater_than_2), std::invalid_argument);

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
  AMG::Jacobi<double> jacobi_base(tolerance, compute_error_every_n_iters,
                                  niters);
  AMG::SuccessiveOverRelaxation<double> sor_base(
      tolerance, compute_error_every_n_iters, niters);

  // Valid multigrid instantiation
  size_t n_fine_nodes = 50;
  size_t n_levels = 4;
  size_t smoothing_iterations = 2;
  AMG::SuccessiveOverRelaxation<double> amg_sor(smoothing_iterations);
  AMG::Multigrid<double> amg(&amg_sor, n_fine_nodes, n_levels);

  // Check coarsening of multigrid system info
  std::cout << "Multigrid System Info:" << std::endl;
  for (size_t level = 1; level < n_levels; ++level) {
    size_t finer_n_nodes = amg.get_n_nodes(level - 1);
    size_t coarser_n_nodes = amg.get_n_nodes(level);
    std::cout << finer_n_nodes << " " << coarser_n_nodes << std::endl;

    size_t finer_n_dofs = amg.get_n_dofs(level - 1);
    size_t coarser_n_dofs = amg.get_n_dofs(level);
    std::cout << finer_n_dofs << " " << coarser_n_dofs << std::endl;

    auto finer_grid_spacing = amg.get_grid_spacing(level - 1);
    auto coarser_grid_spacing = amg.get_grid_spacing(level);
    std::cout << finer_grid_spacing << " " << coarser_grid_spacing << std::endl;

    CHECK(finer_n_nodes > coarser_n_nodes);
    CHECK(finer_n_dofs > coarser_n_dofs);
    CHECK(finer_grid_spacing < coarser_grid_spacing);
  }

  // Check coarsening of multigrid linear systems
  for (size_t level = 1; level < n_levels; ++level) {
    auto finer_A = amg.get_coefficient_matrix(level - 1);
    auto coarser_A = amg.get_coefficient_matrix(level);

    auto finer_u = amg.get_soln(level - 1);
    auto coarser_u = amg.get_soln(level);

    auto finer_b = amg.get_rhs(level - 1);
    auto coarser_b = amg.get_rhs(level);

    CHECK(finer_A.size() > coarser_A.size());
    CHECK(finer_u.size() > coarser_u.size());
    CHECK(finer_b.size() > coarser_b.size());
  }

  // multigrid solution matches direct method
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> ref_direct_solver;
  Eigen::VectorXd ref_exact_u(ndofs);
  auto finest_A = amg.get_coefficient_matrix(0);
  auto finest_b = amg.get_rhs(0);
  ref_direct_solver.analyzePattern(finest_A);
  ref_direct_solver.factorize(finest_A);
  ref_exact_u = ref_direct_solver.solve(finest_b);
  auto amg_u = amg.solve();
  //amg.vcycle();
  // auto amg_u = amg.get_soln(0);

  CHECK(amg_u.isApprox(ref_exact_u, amg.get_tolerance()));
}
