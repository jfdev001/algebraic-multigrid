/**
 * @brief Base class for linear solvers of the form Ax = b
 * 
 * This may not be necessary to define because Eigen already defines other solvers...
 * could also inherit from IterativeSolversBase?
 */
class Solver
{
private:
public:
    Solver();
    ~Solver();
};
