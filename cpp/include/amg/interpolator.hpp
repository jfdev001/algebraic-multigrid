#include <Eigen/Sparse>
#include <vector>

/**
 * @brief Interface for restriction and prolongation operators.
 * 
 */
template <class EleType>
class Interpolator {
 private:
  // LINEAR OPERATORS
  /**
   * @brief 
   * 
   */
  std::vector<Eigen::SparseMatrix<EleType>> restriction;

  /**
   * @brief 
   * 
   */
  std::vector<Eigen::SparseMatrix<EleType>> prolongation;

  EleType theta{0.25};

  /**
   * @brief Defines strength of connection matrices.
   * 
   * References:
   * 
   * [1] : https://github.com/pyamg/pyamg/blob/main/pyamg/strength.py
   * 
   * @param A 
   * @param S 
   * @param T 
   */
  void classical_strength(const Eigen::SparseMatrix<EleType>& A,
                          Eigen::SparseMatrix<EleType>& S,
                          Eigen::SparseMatrix<EleType>& T) {
    int n = A.cols();
    int row; 
    int col;
    EleType _m;
    EleType threshold;
    EleType val;
    Eigen::SparseMatrix<EleType> T{A};
    for (int i = 0; i < n; ++i) {
      _m = find_max_off_diag(A, i);
      threshold = theta * _m;
      for (typename Eigen::SparseMatrix<EleType>::Iterator it(A, i); it; ++it) {
        row = it.row();
        col = it.row();
        val = it.value();
        if (row != i) {
            if (std::abs(val) >= threshold) {
                // update 
                T.insert(row, col) = std::abs(val);
            } else {
                // set zero
                T.insert(row, col) = std::abs(val);
            }
        }
      }
    }

    auto nnz_prev = T.nonZeros();
    T.prune(0.0); // inplace?
    auto nnz_post = T.nonZeros();
    // TODO: remove 
    std::cout << nnz_prev - nnz_post << " <-- dropped zeros" << std::endl;

    scale_cols_by_largest_entry(T);

    return;
  }

  EleType find_max_off_diag(const Eigen::SparseMatrix<EleType>& A, int i) {
    EleType m = 0;
    int row;
    EleType val;
    for (typename Eigen::SparseMatrix<EleType>::Iterator it(A, i); it; ++it) {
        row = it.row();
        val = it.value();
        if (row != i) {
            m = std::max(m, std::abs(val));
        }
    }
    return m;
  }

  EleType find_max(const Eigen::SparseMatrix<EleType>& A, int i) {
    EleType m;
    int row;
    EleType val;
    for (typename Eigen::SparseMatrix<EleType>::Iterator it(A, i); it; ++it) {
        row = it.row();
        val = it.value();
        m = std::max(m, val);
    }
    return m;
  }

  void scale_cols_by_largest_entry(Eigen::SparseMatrix<EleType>& A) { 
    EleType _m;
    int row;
    int col;
    for (int i = 0; i < A.outerSize(); ++i) {
        _m = find_max(A, i);
        for (typename Eigen::SparseMatrix<EleType>::Iterator it(A, i); it; ++it) {
            row = it.row();
            col = it.col();
            A.insert(row, col) = A.coeff(row, col) / _m;
        }
    }
    return; 
  }

 public:
  /**
 * 
*/
  Interpolator(/* args */) {}

  /**
 * 
*/
  ~Interpolator() = default;
};
