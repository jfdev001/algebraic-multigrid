#define CATCH_CONFIG_MAIN
#include <catch2/catch.hpp>

#include <iostream>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/SparseCholesky>

#include <amg/common.hpp>
#include <amg/grid.hpp>
#include <amg/interpolator.hpp>
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

  // Smoothers niters
  size_t niters = 100;

  // Check Jacobi smoother matches exact solution
  Eigen::VectorXd jacobi_u(ndofs);
  jacobi_u.setZero();
  AMG::Jacobi<double> jacobi(niters);
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
  AMG::SuccessiveOverRelaxation<double> sor(niters);
  sor.smooth(A, sor_u, b);
  CHECK(sor_u.isApprox(exact_u, sor.tolerance));

  if (ndofs < 10) {
    std::cout << "BEGIN SOR solution:\n";
    std::cout << sor_u << std::endl;
    std::cout << "END SOR solution\n";
  }

  // Check sparse gauss seidel smoother on small problem
  AMG::SparseGaussSeidel<double> spgs(niters);
  Eigen::VectorXd spgs_u(ndofs);
  spgs_u.setZero();
  spgs.smooth(A, spgs_u, b);
  CHECK(spgs_u.isApprox(exact_u, spgs.tolerance));

  // Instantiate sor/jacobi smoother using Base constructor
  double tolerance = 1e-10;
  size_t compute_error_every_n_iters = 100;
  AMG::Jacobi<double> jacobi_base(tolerance, compute_error_every_n_iters,
                                  niters);
  AMG::SuccessiveOverRelaxation<double> sor_base(
      tolerance, compute_error_every_n_iters, niters);

  // Inspect linear interpolator
  std::cout << "Linear interpolator:" << std::endl;
  size_t n_levels = 9;
  AMG::LinearInterpolator<double> linear_interpolator(n_levels);
  linear_interpolator.make_operators(7, 3, 0);

  std::cout << "nh = 7, nH = 3:" << std::endl;
  std::cout << linear_interpolator.get_P(0) << std::endl;

  std::cout << "nh = 24, nH = 11:" << std::endl;
  linear_interpolator.make_operators(24, 11, 0);
  std::cout << linear_interpolator.get_P(0) << std::endl;

  // Valid multigrid instantiation
  size_t n_fine_nodes = 35;  // too few fine nodes, presmoother will give soln
  std::cout << "Multigrid instantiation:" << std::endl;
  AMG::SparseGaussSeidel<double> amg_spgs;
  std::cout << "AMG SPGS Fields:"
            << "compute error after: " << amg_spgs.compute_error_every_n_iters
            << std::endl
            << "max iters: " << amg_spgs.n_iters << std::endl
            << "tolerance: " << amg_spgs.tolerance << std::endl;
  AMG::Multigrid<double> amg(&linear_interpolator, &amg_spgs, n_fine_nodes,
                             n_levels, 1e-9, 100, 1000);

  // Check coarsening of multigrid linear systems
  std::cout << "Finer dofs --> Coarser dofs" << std::endl;
  for (size_t level = 1; level < n_levels; ++level) {
    auto finer_A = amg.get_coefficient_matrix(level - 1);
    auto coarser_A = amg.get_coefficient_matrix(level);
    std::cout << finer_A.rows() << " --> " << coarser_A.rows() << std::endl;

    auto finer_u = amg.get_soln(level - 1);
    auto coarser_u = amg.get_soln(level);

    auto finer_b = amg.get_rhs(level - 1);
    auto coarser_b = amg.get_rhs(level);

    CHECK(finer_A.size() > coarser_A.size());
    CHECK(finer_u.size() > coarser_u.size());
    CHECK(finer_b.size() > coarser_b.size());
  }

  // Check sparse gaussian solver on larger problem
  // TODO: this should take longer to converge than multigrid but should have
  // approximately the same answers
  std::cout << "Checking sparse gaussian solver" << std::endl;
  AMG::SparseGaussSeidel<double> realistic_spgs(1e-9, 100, 1000);
  auto A_h = AMG::Grid<double>::laplacian(n_fine_nodes);
  auto rhs_h = AMG::Grid<double>::rhs(n_fine_nodes);
  Eigen::VectorXd u_h(rhs_h.rows());
  u_h.setZero();
  realistic_spgs.smooth(A_h, u_h, rhs_h);

  std::cout << "Calling direct solver against sparse gaussian solver"
            << std::endl;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> realistic_direct_solver;
  Eigen::VectorXd realistic_exact_u(rhs_h.rows());
  realistic_direct_solver.analyzePattern(A_h);
  realistic_direct_solver.factorize(A_h);
  realistic_exact_u = realistic_direct_solver.solve(rhs_h);

  std::cout << "Direct error:" << AMG::rss(A_h, realistic_exact_u, rhs_h)
            << std::endl;
  std::cout << "SPGS error: " << AMG::rss(A_h, u_h, rhs_h) << std::endl;

  // multigrid solution matches sparse iterative method
  auto amg_u = amg.solve();

  std::cout << "AMG error: " << AMG::rss(A_h, amg_u, rhs_h) << std::endl;
}
