# algebraic-multigrid

[![CI](https://github.com/jfdev001/algebraic-multigrid/actions/workflows/ci.yml/badge.svg)](https://github.com/jfdev001/algebraic-multigrid/actions/workflows/ci.yml) [![Static Analysis](https://github.com/jfdev001/algebraic-multigrid/actions/workflows/sca.yml/badge.svg)](https://github.com/jfdev001/algebraic-multigrid/actions/workflows/sca.yml) [![Docs](https://github.com/jfdev001/algebraic-multigrid/actions/workflows/docs.yml/badge.svg)](https://jfdev001.github.io/algebraic-multigrid/)

Algebraic multigrid implementation using C++ and Eigen3. The solution via multigrid vcycling of a finite difference discretized Poisson's equation is used to test correctness. Build support tested for MacOS, Windows, and Ubuntu.

# Configuration

To configure:

```bash
cmake -S . -B build
```

Add `-GNinja` if you have Ninja.

To build:

```bash
cmake --build build
```

To test (`--target` can be written as `-t` in CMake 3.15+):

```bash
cmake --build build --target test
```

To build docs (requires [Doxygen](https://github.com/doxygen/doxygen), output in `build/docs/html`):

```bash
cmake --build build --target docs
```

# Example Output

Run the below to see the outputs of the test, that last line of which shows the error and demonstrates how much more quickly AMG converges relative to a sparse iterative solver:

```
./build/test/testlib
```

![output](image/README/output.png)

# Debugging in `cpp/`

```shell
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug # only do once
cmake --build build-debug
```

# Profiling

For profiling the code, I use KCacheGrind and Callgrind, which can be installed as below,

```shell
sudo apt-get install valgrind kcachegrind graphviz
```

run the program with the `valgrind` program, noting that the program will take longer than normal due to the profiling overhead,

```shell
valgrind --tool=callgrind program [program_options]
```

where `program` is binary from `cmake --build build`. So as an example,

```shell
valgrind --tool=callgrind ./build/test/testlib
```

followed by

```shell
kcachegrind callgrind<tab autcomplete>
```

will allow you to visualize the callgraph and identify performance bottlenecks.

# Notes

## Sparse Gauss Seidel

To keep things generic, one can use either the matrix formulation `Au = b` or one can write solvers that use the physical grid points themselves (see ref [15]). Using the physical grid points in the solvers leads to solvers that are defined only for that particular PDE, so for this project I take the more generic approach in which I develop algorithms explicitly taking the coefficient matrix `A` as input. Defining the iterative methods in terms of the physical grid would make the algorithm a [matrix-free method](https://en.wikipedia.org/wiki/Matrix-free_methods), and while this an efficient approach, it can mean iterative solvers need to be defined explicitly using knowledge of the underyling PDE.

One thing I noticed when working on the smoothers, is that a disproportionate amount of time was being spent here. To make AMG competitive, I needed to improve upon this. The formulations given in numerical linear algebra texts tends to assume dense matrices; however, we know that our input is a sparse matrix. Therefore, I knew I needed to use sparse formulation of the smoothers, and took inspirtation from Julia's [AlgebraicMultigrid/src/smoother.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/blob/master/src/smoother.jl).

![slow_smoother_callgraph](image/README/slow_smoother_callgraph.png)

In the sparse gauss seidel solver, the use of the ternary operator in the sparse matrix vector product arises from the definition of the Gauss Seidel iteration. The Gauss seidel iterations are defined as

$$
u_{i}^{(k+1)} = \frac{b_i - \sum_{j < i} a_{ij} u_j^{(k+1)} - \sum_{j > i} a_{ij} u_j^{(k)} }{a_{ii}},\ i = 1...n,
$$

and this implies that $j \neq i$ in the summations. Therefore the ternary operator for which the row and column match requires that the contribution to the summation be 0. The above might be more easily understood (and removing the $k^{th}$ iteration subscript for ease) as

$$
u_{i}^{(k+1)} = \frac{b_i - \sum_{j \neq i} a_{ij} u_j }{a_{ii}}.
$$

However, since we know that $A$ is sparse, for any given column $j$, we only need a small subset of the rows $i$ from $A$. It's worth noting that since our matrix is CSC format, it is faster to iterate through the column vectors (i.e, the rows) of $A$, and this iteration on the surface contradicts the formula given above, since the formula above is iterating through row vectors (i.e., the columns). However, since we assume $A$ is symmetric positive definite, then we know that $A^{T} = A$ and therefore iterating throughing the $j^{th}$ column vector is the same as iterating through the $j^{th}$ row vector.

## Interpolation

For the restriction and prolongation operation, this seems to be dependent on the selected multigrid method. The [classical.jl from AlgebraicMultigrid.jl](https://github.com/JuliaLinearAlgebra/AlgebraicMultigrid.jl/blob/master/src/classical.jl) shows that this operation is performed using Ruge-Stuben method.

Though a simpler approach, i.e., linear interpolation is what is proposed in refs [9]
and [19], and it is also the approach that I take for ease of implementation.

# References

[1] : Intro Modern CMake. url: https://cliutils.gitlab.io/modern-cmake/chapters/basics/structure.html

[2] : Pawar S, San O. 6.3: Multigrid Framework in "CFD Julia: A Learning Module
Structuring an Introductory Course on Computational Fluid Dynamics". Fluids.
2019; 4(3):159. https://doi.org/10.3390/fluids4030159

[3] : DOLFINX: Python Binding Example for C++. url: https://github.com/FEniCS/dolfinx

[4] : Nanobind docs. url: https://nanobind.readthedocs.io/en/latest/installing.html

[5] : Modern CMake: Simple Example and Links to Extended Examples. url: https://cliutils.gitlab.io/modern-cmake/chapters/basics/example.html

[6] : amgcl: Good Inspiration for AMG design. url: https://amgcl.readthedocs.io/en/latest/amg_overview.html

[7] : AMG = iterative solver if V-cycle until n-iterations or convergence, but more often a single V-cycle used for preconditioner to get (?) $M^{-1} v$ (see BDDC for example, or Preconditioners/diagonal.jl). url: https://github.com/ddemidov/amgcl/issues/230

[8] : Long, Chen. Programming of Multigrid Methods. url: https://www.math.uci.edu/~chenlong/226/MGcode.pdf

[9] : Kostler, Harald. Multigrid HowTo: A simple Multigrid solver in C++ in less
than 200 lines of code. url: https://www10.cs.fau.de/publications/reports/TechRep_2008-03.pdf

[10] : On class template and header files. url: https://stackoverflow.com/questions/495021/why-can-templates-only-be-implemented-in-the-header-file

[11] : Cmake: Header only library. url: https://stackoverflow.com/questions/60604249/how-to-make-a-header-only-library-with-cmake

[12] : Build cmake debugging. url: https://hsf-training.github.io/hsf-training-cmake-webpage/08-debugging/

[13] : For good CI/CD example, see Dolfinx workflow. url: https://github.com/FEniCS/dolfinx/blob/main/.github/workflows/ccpp.yml

[14] : General DevSecOps. url: https://medium.com/@rahulsharan512/devsecops-using-github-actions-building-secure-ci-cd-pipelines-5b6d59acab32

[15] : Iterative solvers and square vs. column formulation. url: https://people.eecs.berkeley.edu/~demmel/cs267/lecture24/lecture24.html

[16] Multigrid in MATLAB with a recursive algorithm. url: https://nl.mathworks.com/help/parallel-computing/solve-differential-equation-using-multigrid-preconditioner-on-distributed-discretization.html

[17] : Ruge, J. W., & Stüben, K. (1987). Algebraic multigrid (AMG). In S. F. McCormick (Ed.), Multigrid methods (Vol. 3, pp. 73–130). SIAM. https://doi.org/10.1137/1.9781611971057

[18] : Classical Ruge-Stuben Multigrid Python Code. url: https://pyamg.readthedocs.io/en/latest/\_modules/pyamg/classical/classical.html

[19] : Briggs et. al. Chapter 4: Implementation in "A Multigrid Tutorial, 2ed" (2000).

[20] : Quetzal C++ Doxygen Workflow Example. url: https://github.com/Quetzal-framework/quetzal-CoaTL/tree/master

[21] : Github Pages Deployment via Github Actions (make sure github pages set to deploy from gh-pages branch). url: https://github.com/peaceiris/actions-gh-pages?tab=readme-ov-file
