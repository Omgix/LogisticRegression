#include <iostream>
#include <numeric>
#include <random>
#include <vector>
#include <omp.h>

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

    const int SIZE = 20;

    for (unsigned k = 0; k < n_samples; k += SIZE) {
      std::vector<Eigen::SparseVector<double>> gradients;
      for (unsigned i = 0; i < SIZE; ++i)
        gradients.emplace_back(n_features);
      
      #pragma omp parallel for schedule(dynamic)
      for (unsigned int j = k; j < k + SIZE; j++){
        if (j < n_samples) {
          double alpha = _alpha / sqrt(j + n_samples * n + 1);
          mu += (_l2*_alpha);
          int label = target[index[k]];
          SpMat::InnerIterator iter = SpMat::InnerIterator(samples, index[k]);
          double logit = 0;
          for(auto it = iter; it; ++it)
            logit += it.value() * _weights[it.col()];
          double predicted = sigmoid(logit);
          for(auto it = iter; it; ++it){
            if(_l2)
              gradients[j-k].coeffRef(it.col()) += alpha * ((label - predicted) * it.value() + 2 * _l2 * _weights[it.col()]);
            else
              gradients[j-k].coeffRef(it.col()) += alpha * (label - predicted) * it.value();
          }
        }
      }

      #pragma omp parallel for schedule(static)
      for (unsigned i = 0; i < SIZE; ++i)
        _weights += gradients[i];
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