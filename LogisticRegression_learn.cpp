#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "LogisticRegression.h"

bool
LogisticRegression::learn(const SpMat &samples, const Vector &target, bool verbose) {
  if (samples.rows() != target.size()) {
    _error_msg = "Unmached rows of sample data and size of target vector";
    return false;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  long n_samples = samples.rows();
  long n_features = samples.cols();
  _weights = Vector::Zero(n_features);
  std::vector<int> index(n_samples);
  std::iota(index.begin(),index.end(),0);
  Vector total_l2 = Vector::Zero(n_features);

  double mu = 0.0;
  double norm = 1.0;
  unsigned int n = 0;

  if (verbose)
    std::cout << "# stochastic gradient descent" << std::endl;
  while(norm > _eps){

    Eigen::VectorXd old_weights(_weights);
    if(_shuf)
      std::shuffle(index.begin(),index.end(),g);

    for (unsigned int i = 0; i < n_samples; i++){
      mu += (_l2*_alpha);
      double alpha = _alpha / sqrt(i + n_samples * n + 1);
      int label = target[index[i]];
      SpMat::InnerIterator iter = SpMat::InnerIterator(samples, index[i]);
      double logit = 0;
      for(auto it = iter; it; ++it)
        logit += it.value() * _weights[it.col()];
      double predicted = sigmoid(logit);
      for(auto it = iter; it; ++it){
        if(_l2)
          _weights[it.col()] += alpha * ((label - predicted) * it.value() + 2 * _l2 * _weights[it.col()]);
        else
          _weights[it.col()] += alpha * (label - predicted) * it.value();
      }
    }
    norm = (_weights - old_weights).norm();
    if(n && n % 10 == 0){
      double l2n = _weights.norm();
      if (verbose)
        printf("# convergence: %1.4f l2-norm: %1.4e iterations: %i\n",norm,l2n,n);
    }
    if(++n > _maxit){
      _error_msg = "Max iterations reached";
      return false;
    }
  }
  return true;
}