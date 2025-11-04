#include <iostream>
#include <cmath>

template<typename T>
void initilisation(const T* A, const T* b, T* r, T* P, const T* x0, int size) {
  for(int i = 0; i < size; ++i) {
    r[i] = 0.0;
    for(int j = 0; j < size; ++j) {
      r[i] += A[i*size + j] * x0[j];
    }
    r[i] -= b[i];
    P[i] = -r[i];
  }
}

template<typename T>
T compute_alpha(const T* A, const T* P, T* AP, 
  const T* r, int size, int k, T& alpha_num) {

  T  alpha_denom = 0.0f;

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

template<typename T>
T compute_beta(const T* P, const T* AP, 
  T* r, T* x, int size, T& alpha_num, T alpha) {

  T old_alpha_num = alpha_num;
  alpha_num = 0.0f;

  for (int i = 0; i < size; ++i) {
    x[i] = x[i] + alpha * P[i];
    r[i] = r[i] + alpha * AP[i];
    alpha_num += r[i] * r[i];
  }

  return alpha_num / old_alpha_num;
}

template<typename T>
bool compute_P(T* P, const T* r, T beta, int size, T tol, int k){
  bool stay_condition = false;
  for(int i = 0; i < size; ++i) {
    P[i] = -r[i] + beta * P[i];
    if (std::fabs(r[i]) > tol) stay_condition = true;
  }

  // if (size < k) stay_condition = false;
  return stay_condition;
}

template<typename T>
void Conjugate_Gradien(const T* A, const T* b, T* x0, int state, int Knot_point, T tol = 1e-6) {
  
  if (!A || !b || !x0) {
    std::cerr << "error: A, b or x0 is nullptr\n";
    return;
  }

  int size = state * Knot_point;
  T* r  = new T[size];
  T* P  = new T[size];
  T* AP = new T[size];
  initilisation(A, b, r, P, x0, size);
  int k = 0;
  T alpha_num = 0.0f, alpha = 0.0f, beta = 0.0f;
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

