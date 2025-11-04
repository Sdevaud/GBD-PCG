#include <iostream>
#include <vector>
#include <iomanip> // pour std::setprecision

using namespace std;

template<typename T>
void printVector(string vector_name, const T* arr, int size, int precision = 4) {
  cout << vector_name << " : ";  
  cout << "[";
    for (int i = 0; i < size; i++) {
        cout << fixed << setprecision(precision) << arr[i];
        if (i != size - 1) cout << ", ";
    }
    cout << "]" << endl;
}

template<typename T>
void printMatrix(string matrix_name, const T* mat, int size, int precision = 4) {
  cout << matrix_name << " : \n";
  for (int i = 0; i < size; i++) {
      cout << "[";
      for (int j = 0; j < size; j++) {
          cout << fixed << setprecision(precision) << mat[i * size + j];
          if (j != size - 1) cout << ", ";
      }
      cout << "]" << endl;
  }
}

template<typename T>
void mat_mul_vector(const T* matrix, const T* vector, T* result, int size) {
  for (int i = 0; i < size; ++i) {
    for(int j = 0; j < size; ++j) {
      result[i] += matrix[i*size + j] * vector[j];
    }
  }
}