# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"

int main (int argc, char **argv) {
  // reading cmd line arguments
  int n = atoi(argv[1]);

  // creating cublas handle
  cudaError_t cudaStat ;
  cublasStatus_t stat ;
  cublasHandle_t handle ;
  stat = cublasCreate(& handle);

  // allocating memory for vectors on host
  float * x;
  float * y;
  x = (float *) malloc(n * sizeof (*x));
  y = (float *) malloc(n * sizeof (*y));

  // setting up values in vectors
  for (int j = 0; j < n; j++) {
    x[j] = (float)j;
  }
  for (int j = 0; j < n; j++) {
    y[j] = (float)j;
  }

  printf ("Original vector x:\n");
  for (int j = 0; j < n; j++) {
    printf ("%2.0f, ", x[j]);
  }
  printf ("\n");
  for (int j = 0; j < n; j++) {
    printf ("%2.0f, ", y[j]);
  }
  printf ("\n");

  // using cudamalloc for allocating memory on device
  float * d_x;
  float * d_y;
  cudaStat = cudaMalloc(( void **)& d_x, n* sizeof (*x));
  cudaStat = cudaMalloc(( void **)& d_y, n* sizeof (*y));

  // setting values of matrices on device
  stat = cublasSetVector(n, sizeof (*x), x, 1, d_x, 1);
  stat = cublasSetVector(n, sizeof (*y), y, 1, d_y, 1);

  // scalar quantity to be multiplied with x
  float al = 2.0;

  // performing saxpy operation
  stat = cublasSaxpy(handle, n, &al, d_x, 1, d_y, 1);

  // getting the final output
  stat = cublasGetVector(n, sizeof ( float ), d_y, 1, y, 1);

  // final output
  printf ("Final output y after Saxpy operation:\n");
  for (int j = 0; j < n; j++) {
    printf ("%2.0f, ", y[j]);
  }
  printf ("\n");

  // free device memory
  cudaFree(d_x);
  cudaFree(d_y);

  // destroying cublas handle
  cublasDestroy(handle);

  // freeing host memory
  free(x);
  free(y);

  return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// y after Saxpy :
// 0 , 3 , 6 , 9 ,12 ,15 ,// 2*x+y = 2*{0 ,1 ,2 ,3 ,4 ,5} + {0 ,1 ,2 ,3 ,4 ,5}
