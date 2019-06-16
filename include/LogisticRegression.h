#ifndef LOGISTIC_REGRESSION_SGD_LOGISTICREGRESSION_HPP
#define LOGISTIC_REGRESSION_SGD_LOGISTICREGRESSION_HPP

#include <Eigen/Sparse>
#include <fstream>
#include <string>

class LogisticRegression {
 public:
  typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
  typedef Eigen::VectorXd Vector;
  explicit LogisticRegression(bool shuf = true, double alpha = 0.001, double l1 = 0.0001,
      double eps = 0.005, unsigned maxit = 50000);
  bool learn(const SpMat &samples, const Vector &target, bool verbose = false);
  long features() { return _weights.size(); }
  long sparse_features();
  double predict(Vector &data);
  void write_weight(std::ofstream &outfile) {
    for (int i = 0; i < _weights.size(); ++i)
      outfile << i << " " << _weights[i] << std::endl;
  }
 private:
  Vector _weights;
  double _alpha;
  double _l1;
  double _eps;
  unsigned _maxit;
  bool _shuf;
  std::string _error_msg;
  static double sigmoid(double x);
};

#endif //LOGISTIC_REGRESSION_SGD_LOGISTICREGRESSION_HPP
