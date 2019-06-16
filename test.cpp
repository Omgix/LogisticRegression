// L1-regularized logistic regression implementation using stochastic gradient descent
// (c) Tim Nugent
// timnugent@gmail.com

#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <sstream>
#include <numeric>
#include <algorithm>
#include <map>

#include <Eigen/Sparse>
#include "LogisticRegression.h"

using namespace std;

typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
typedef Eigen::Triplet<double> Triplet;

vector<string> split(const string &s, char delim, vector<string> &elems) {
  stringstream ss(s);
  string item;
  while (getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

vector<string> split(const string &s, char delim) {
  vector<string> elems;
  split(s, delim, elems);
  return elems;
}

void usage(const char* prog){

   cout << "Read training data then classify test data using logistic regression:\nUsage:\n"
    << prog << " [options] [training_data]" << endl << endl;
   cout << "Options:" << endl;   
   cout << "-s <int>   Shuffle dataset after each iteration. default 1" << endl;  
   cout << "-i <int>   Maximum iterations. default 50000" << endl;   
   cout << "-e <float> Convergence rate. default 0.005" << endl;  
   cout << "-a <float> Learning rate. default 0.001" << endl; 
   cout << "-l <float> L1 regularization weight. default 0.0001" << endl; 
   cout << "-m <file>  Read weights from file" << endl;  
   cout << "-o <file>  Write weights to file" << endl;   
   cout << "-t <file>  Test file to classify" << endl;   
   cout << "-p <file>  Write predictions to file" << endl;   
   cout << "-r     Randomise weights between -1 and 1, otherwise 0" << endl;  
   cout << "-v     Verbose." << endl << endl;    
}


double sigmoid(double x){

  static double overflow = 20.0;
  if (x > overflow) x = overflow;
  if (x < -overflow) x = -overflow;

  return 1.0/(1.0 + exp(-x));
}

double classify(SpMat::InnerIterator& iter, Eigen::VectorXd& weights){

  double logit = 0.0;
  for(auto it = iter; it; ++it)
    logit += it.value() * weights[it.col()];

  return sigmoid(logit);
}


int main(int argc, const char* argv[]){

  // Learning rate
  double alpha = 0.001;
  // L1 penalty weight
  double l1 = 0.0001;
  // Max iterations
  unsigned int maxit = 50000;
  // Shuffle data set
  int shuf = 1;
  // Convergence threshold
  double eps = 0.005;
  // Verbose
  int verbose = 0;
  // Randomise weights
  int randw = 0;
  // Read model file
  string model_in = "";
  // Write model file
  string model_out = "";
  // Test file
  string test_file = "";   
  // Predictions file
  string predict_file = "";

  if(argc < 2){
    usage(argv[0]);
    return(1);
  }else{
    cout << "# called with:     ";
    for(int i = 0; i < argc; i++){
      cout << argv[i] << " ";
      if(string(argv[i]) == "-a" && i < argc-1){
        alpha = atof(argv[i+1]);
      }
      if(string(argv[i]) == "-m" && i < argc-1){
        model_in = string(argv[i+1]);
      }
      if(string(argv[i]) == "-o" && i < argc-1){
        model_out = string(argv[i+1]);
      }
      if(string(argv[i]) == "-t" && i < argc-1){
        test_file = string(argv[i+1]);
      }
      if(string(argv[i]) == "-p" && i < argc-1){
        predict_file = string(argv[i+1]);
      }
      if(string(argv[i]) == "-s" && i < argc-1){
        shuf = atoi(argv[i+1]);
      }
      if(string(argv[i]) == "-i" && i < argc-1){
        maxit = atoi(argv[i+1]);
      }
      if(string(argv[i]) == "-e" && i < argc-1){
        eps = atof(argv[i+1]);
      }
      if(string(argv[i]) == "-l" && i < argc-1){
        l1 = atof(argv[i+1]);
      }
      if(string(argv[i]) == "-v"){
        verbose = 1;
      }
      if(string(argv[i]) == "-r"){
        randw = 1;
      }
      if(string(argv[i]) == "-h"){
        usage(argv[0]);
        return(1);
      }
    }
    cout << endl;
  }
 
  if(!model_in.length()){
    cout << "# learning rate:   " << alpha << endl;
    cout << "# convergence rate:  " << eps << endl;
    cout << "# l1 penalty weight: " << l1 << endl;
    cout << "# max. iterations:   " << maxit << endl;   
    cout << "# training data:   " << argv[argc-1] << endl;
    if(model_out.length()) cout << "# model output:    " << model_out << endl;
  }
  if(model_in.length()) cout << "# model input:     " << model_in << endl;
  if(test_file.length()) cout << "# test data:     " << test_file << endl;
  if(predict_file.length()) cout << "# predictions:     " << predict_file << endl;

  std::vector<Triplet> data;
  Eigen::VectorXd weights;
  Eigen::VectorXd target;
  random_device rd;
  mt19937 g(rd());
  ifstream fin;
  string line;
  size_t n_features = 0;
  size_t n_samples = 0;

  // Read weights from model file, if provided
  if(model_in.length()){
    fin.open(model_in.c_str());
    size_t idx = 0;
    while (getline(fin, line)){
      if(line.length()){
        if(line[0] != '#' && line[0] != ' '){
          vector<string> tokens = split(line,' ');
          if(tokens.size() == 2){
            weights[atoi(tokens[0].c_str())] = atof(tokens[1].c_str());
            idx++;
          }
        } else {
          size_t pos = line.find("SIZE");
          if (pos != std::string::npos) {
            stringstream sizess (line.substr(pos + 4));
            sizess >> n_features;
            weights = Eigen::VectorXd::Zero(n_features);
            target = Eigen::VectorXd::Zero(n_features);
          }
        }
      }
    }
    if(!n_features){
      cout << "# failed to read weights from file!" << endl;
      fin.close();    
      exit(-1);
    }fin.close();
  }

  LogisticRegression model (shuf, alpha, l1, eps, maxit);

  // If no weights file provided, read training file and calculate weights
  if(!weights.size()){

    fin.open(argv[argc-1]);
    int idx = 0;
    while (getline(fin, line)){
      if(line.length()){
        if(line[0] != '#' && line[0] != ' '){
          vector<string> tokens = split(line,' ');
          if(atoi(tokens[0].c_str()) == 1){
            target[idx] = 1;
          }else{
            target[idx] = 0;
          }
          for(unsigned int i = 1; i < tokens.size(); i++){
            //if(strstr (tokens[i],"#") == NULL){
              vector<string> feat_val = split(tokens[i],':');
              if(feat_val.size() == 2){
                data.emplace_back(idx, atoi(feat_val[0].c_str()), atof(feat_val[1].c_str()));
                if(randw){
                  weights[atoi(feat_val[0].c_str())] = -1.0+2.0*(double)rd()/rd.max();
                }else{
                  weights[atoi(feat_val[0].c_str())] = 0.0;
                }
              }
            //}
          }
          idx++;
          //if(verbose) cout << "read example " << data.size() << " - found " << example.size()-1 << " features." << endl; 
        } else {
          size_t pos = line.find("SIZE");
          if (pos != std::string::npos) {
            size_t comma = line.find(',');
            stringstream sizess (line.substr(pos + 4, comma));
            if (comma != std::string::npos) {
              sizess >> n_samples;
              stringstream featuresss (line.substr(comma + 1));
              featuresss >> n_features;
              weights = Eigen::VectorXd::Zero(n_features);
              target = Eigen::VectorXd::Zero(n_features);
            }
          }
        }  
      }
    }
    fin.close();

    cout << "# training examples: " << n_samples << endl;
    cout << "# features:      " << weights.size() << endl;

    SpMat samples = SpMat(n_samples, n_features);
    samples.setFromTriplets(data.begin(), data.end());

    auto start = std::chrono::steady_clock::now();
    model.learn(samples, target, true);
    auto end =  std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "# time to convergence: " << (double)duration.count() / 1000 << 's' << std::endl;

    unsigned int sparsity = model.sparse_features();
    printf("# sparsity:  %1.4f (%i/%i)\n",(double)sparsity/weights.size(),sparsity,(int)weights.size());

    if(model_out.length()){
      ofstream outfile;
      outfile.open(model_out.c_str());
      model.write_weight(outfile);
      cout << "# written weights to file " << model_out << endl;
    }

  }

  // If a test file is provided, classify it using either weights from
  // the provided weights file, or those just calculated from training
  if(test_file.length()){

    ofstream outfile;
    if(predict_file.length()){
      outfile.open(predict_file.c_str());  
    }

    cout << "# classifying" << endl;
    double tp = 0.0, fp = 0.0, tn = 0.0, fn = 0.0;
    fin.open(test_file.c_str());
    while (getline(fin, line)){
      if(line.length()){
        if(line[0] != '#' && line[0] != ' '){
          vector<string> tokens = split(line,' ');
          Eigen::VectorXd example = Eigen::VectorXd::Zero(n_features);
          int label = atoi(tokens[0].c_str());
          for(unsigned int i = 1; i < tokens.size(); i++){
            vector<string> feat_val = split(tokens[i],':');
            example[atoi(feat_val[0].c_str())] = atof(feat_val[1].c_str());
          }
          double predicted = model.predict(example);
          if(verbose){
            if(label > 0){
              printf("label: +%i : prediction: %1.3f",label,predicted);
            }else{
              printf("label: %i : prediction: %1.3f",label,predicted);
            }
          }
          if(predict_file.length()){
            if(predicted >= 0.5){
              outfile << "1" << endl;
            }else{
              outfile << "0" << endl;
            }
          }
          if(((label == -1 || label == 0) && predicted < 0.5) || (label == 1 && predicted >= 0.5)){
            if(label == 1){tp++;}else{tn++;}  
            if(verbose) cout << "\tcorrect" << endl;
          }else{
            if(label == 1){fn++;}else{fp++;}  
            if(verbose) cout << "\tincorrect" << endl;
          }
        } else {
          size_t pos = line.find("SIZE");
          if (pos != std::string::npos) {
            size_t comma = line.find(',');
            stringstream sizess (line.substr(pos + 4, comma));
            if (comma != std::string::npos) {
              sizess >> n_samples;
              stringstream featuresss (line.substr(comma + 1));
              size_t len_feature = n_features;
              featuresss >> n_features;
              if (n_features != len_feature) {
                cout << "Unmached number of features: (" << n_features <<'/' << len_feature << ')' << endl;
                break;
              }
            }
          }
        }
      }
    }
    fin.close();

    printf ("# accuracy:  %1.4f (%i/%i)\n",((tp+tn)/(tp+tn+fp+fn)),(int)(tp+tn),(int)(tp+tn+fp+fn));
    printf ("# precision:   %1.4f\n",tp/(tp+fp));
    printf ("# recall:    %1.4f\n",tp/(tp+fn));
    printf ("# mcc:     %1.4f\n",((tp*tn)-(fp*fn))/sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)));
    printf ("# tp:      %i\n",(int)tp);
    printf ("# tn:      %i\n",(int)tn);
    printf ("# fp:      %i\n",(int)fp);  
    printf ("# fn:      %i\n",(int)fn);

    if(predict_file.length()){
      cout << "# written predictions to file " << predict_file << endl;
      outfile.close();
    }  
  }

  return(0);

}
