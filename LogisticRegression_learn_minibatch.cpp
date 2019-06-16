#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <omp.h>

#include "LogisticRegression.h"

bool
LogisticRegression::learn(const SpMat &samples, const Vector &target, bool verbose) {
  if (samples.cols() != target.size()) {
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
  Vector total_l1 = Vector::Zero(n_features);

  double mu = 0.0;
  double norm = 1.0;
  unsigned int n = 0;

  if (verbose)
    std::cout << "# stochastic gradient descent" << std::endl;
  while(norm > _eps){

    Eigen::VectorXd old_weights(_weights);
    if(_shuf)
      std::shuffle(index.begin(),index.end(),g);

    const int SIZE = 4;

    for (unsigned k = 0; k < n_samples; k += SIZE) {
      std::vector<Eigen::SparseVector<double>> gradients;
      for (unsigned i = 0; i < SIZE; ++i)
        gradients.emplace_back(n_features);
      
      #pragma omp parallel for schedule(dynamic)
      for (unsigned int j = k; j < k + SIZE; j++){
        if (j < n_samples) {
          mu += (_l1*_alpha);
          int label = target[index[k]];
          SpMat::InnerIterator iter = SpMat::InnerIterator(samples, index[k]);
          double logit = 0;
          for(auto it = iter; it; ++it)
            logit += it.value() * _weights[it.col()];
          double predicted = sigmoid(logit);
          for(auto it = iter; it; ++it){
            gradients[j-k].coeffRef(it.col()) += _alpha * (label - predicted) * it.value();
            if(_l1){
              // Cumulative L1-regularization
              // Tsuruoka, Y., Tsujii, J., and Ananiadou, S., 2009
              // http://aclweb.org/anthology/P/P09/P09-1054.pdf
              double z = _weights[it.col()];
              if(_weights[it.col()] > 0.0){
                _weights[it.col()] = std::max(0.0,(double)(_weights[it.col()] - (mu + total_l1[it.col()])));
              }else if(_weights[it.col()] < 0.0){
                _weights[it.col()] = std::min(0.0,(double)(_weights[it.col()] + (mu - total_l1[it.col()])));
              }
              total_l1[it.col()] += (_weights[it.col()] - z);
            }
          }
        }
      }

      #pragma omp parallel for schedule(static)
      for (unsigned i = 0; i < SIZE; ++i)
        _weights += gradients[i];
    }

    norm = (_weights - old_weights).norm();
    if(n && n % 100 == 0){
      double l1n = _weights.cwiseAbs().sum();
      if (verbose)
        printf("# convergence: %1.4f l1-norm: %1.4e iterations: %i\n",norm,l1n,n);
    }
    if(++n > _maxit){
      _error_msg = "Max iterations reached";
      return false;
    }
  }
  return true;
}