# algebraic-multigrid
Algebraic multigrid implementation using C++ and Eigen

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
