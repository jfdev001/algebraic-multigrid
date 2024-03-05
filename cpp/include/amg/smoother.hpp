/**
 * @brief Base class for linear Smoothers of the form Ax = b
 * 
 * This may not be necessary to define because Eigen already defines other Smoothers...
 * could also inherit from IterativeSmoothersBase?
 */
class Smoother
{
private:
public:
    Smoother();
    ~Smoother();
};
