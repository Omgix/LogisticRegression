#include <cmath>

#include "include/LogisticRegression.h"

LogisticRegression::LogisticRegression(bool shuf, double alpha, double l2, double eps, unsigned int maxit):
                                      _alpha(alpha), _l2(l2), _eps(eps), _maxit(maxit), _shuf(shuf){}


long
LogisticRegression::sparse_features() {
  long sparsity = 0;
  for (int i = 0; i < _weights.size(); ++i)
    if (_weights[i] != 0)
      sparsity++;

  return sparsity;
}

double
LogisticRegression::predict(Vector &data) {
  return sigmoid(data.dot(_weights));
}

double
LogisticRegression::sigmoid(double x)
{
  static double overflow = 20.0;
  if (x > overflow) x = overflow;
  if (x < -overflow) x = -overflow;

  return 1.0/(1.0 + exp(-x));
}