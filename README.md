# algebraic-multigrid

Algebraic multigrid implementation using C++ and Eigen.

## Configuration in `cpp/`

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

To build docs (requires Doxygen, output in `build/docs/html`):

```bash
cmake --build build --target docs
```

# Classes

Planning for classes needed for algebraic multigrid.

```cpp
BaseSolver {
private:
  niters
  tolerance
  (?) Preconditioner
public:
  abstract solve(const Matrix& A, Vector& x, const Vector& b) const // inplace vector sys solve
  abstract solve(const Matrix& A, Matrix& X, const Matrix& B) const // inplace matrix sys solve
}

BaseGrid {
private:
  Matrix pdeSolution
  Matrix error
  Matrix residual
  dx
  dy
  nx
  ny
public:
  const& getters const
}

Multigrid {
private:
  BaseGrid[] grids # 0th == level 1 finest --> n-1^th == coarsest 
  nlevels
  Int[] iters
  restrict()
  prolongate()
public:
  solve()
}
```

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
