#include <iostream>
#include <cmath>

void initilisation(const double* A, const double* b, double* r, double* P, const double* x0, int size) {
  for(int i = 0; i < size; ++i) {
    r[i] = 0.0;
    for(int j = 0; j < size; ++j) {
      r[i] += A[i*size + j] * x0[j];
    }
    r[i] -= b[i];
    P[i] = -r[i];
  }
}

double compute_alpha(const double* A, const double* P, double* AP, 
  const double* r, int size, int k, double& alpha_num) {

  double  alpha_denom = 0.0f;

  for (int i = 0; i < size; ++i) {
    AP[i] = 0.0f;
    for (int j = 0; j < size; ++j) {
      alpha_denom += P[i] * A[i*size +j] * P[j];
      AP[i] += A[i*size +j] * P[j];
    }
    if(k==0) alpha_num += r[i] * r[i];
  }



  return alpha_num / alpha_denom;
}

double compute_beta(const double* P, const double* AP, 
  double* r, double* x, int size, double& alpha_num, double alpha) {

  double old_alpha_num = alpha_num;
  alpha_num = 0.0f;

  for (int i = 0; i < size; ++i) {
    x[i] = x[i] + alpha * P[i];
    r[i] = r[i] + alpha * AP[i];
    alpha_num += r[i] * r[i];
  }

  

  return alpha_num / old_alpha_num;
}

bool compute_P(double* P, const double* r, double beta, int size, double tol, int k){
  bool stay_condition = false;
  for(int i = 0; i < size; ++i) {
    P[i] = -r[i] + beta * P[i];
    if (std::fabs(r[i]) > tol) stay_condition = true;
  }

  // if (size < k) stay_condition = false;
  return stay_condition;
}


void Conjugate_Gradien(const double* A, const double* b, double* x0, int state, int Knot_point, double tol = 1e-6) {
  if (!A || !b || !x0) {
    std::cerr << "error: A, b or x0 is nullptr\n";
    return ;
}
  int size = state * Knot_point;
  double* r  = new double[size];
  double* P  = new double[size];
  double* AP = new double[size];
  initilisation(A, b, r, P, x0, size);
  int k = 0;
  double alpha_num = 0.0f, alpha = 0.0f, beta = 0.0f;
  bool stay_condition = true;
  

  do {
    alpha = compute_alpha(A, P, AP, r, size, k, alpha_num);
    beta = compute_beta (P, AP, r, x0, size, alpha_num, alpha);
    stay_condition = compute_P(P, r, beta, size, tol, k);
    ++k;
  } while(stay_condition);

  delete[] r;
  delete[] P;
  delete[] AP;

}

