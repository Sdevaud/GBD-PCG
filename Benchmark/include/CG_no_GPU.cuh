#include <iostream>
#include <cmath>

void initilisation(const float* A, const float* b, float* r, float* P, const float* x0, int size) {
  for(int i = 0; i < size; ++i) {
    r[i] = 0.0;
    for(int j = 0; j < size; ++j) {
      r[i] += A[i*size + j] * x0[j];
    }
    r[i] -= b[i];
    P[i] = -r[i];
  }
}

float compute_alpha(const float* A, const float* P, float* AP, 
  const float* r, int size, int k, float& alpha_num) {

  float  alpha_denom = 0.0f;

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

float compute_beta(const float* P, const float* AP, 
  float* r, float* x, int size, float& alpha_num, float alpha) {

  float old_alpha_num = alpha_num;
  alpha_num = 0.0f;

  for (int i = 0; i < size; ++i) {
    x[i] = x[i] + alpha * P[i];
    r[i] = r[i] + alpha * AP[i];
    alpha_num += r[i] * r[i];
  }

  

  return alpha_num / old_alpha_num;
}

bool compute_P(float* P, const float* r, float beta, int size, float tol, int k){
  bool stay_condition = false;
  for(int i = 0; i < size; ++i) {
    P[i] = -r[i] + beta * P[i];
    if (std::fabs(r[i]) > tol) stay_condition = true;
  }

  // if (size < k) stay_condition = false;
  return stay_condition;
}


void Conjugate_Gradien(const float* A, const float* b, float* x0, int state, int Knot_point, float tol = 1e-6) {
  if (!A || !b || !x0) {
    std::cerr << "error: A, b or x0 is nullptr\n";
    return ;
}
  int size = state * Knot_point;
  float* r  = new float[size];
  float* P  = new float[size];
  float* AP = new float[size];
  initilisation(A, b, r, P, x0, size);
  int k = 0;
  float alpha_num = 0.0f, alpha = 0.0f, beta = 0.0f;
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

