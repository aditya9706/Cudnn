/**
 * Copyright 2021-2022 Enflame. All Rights Reserved.
 *
 * @file    cublas_saxpy_test.cpp
 * @brief   Benchmarking Tests for cublas Saxpy API
 *
 * @author  ashish(CAI)
 * @date    2021-12-17
 * @version V1.0
 * @par     Copyright (c)
 *          Enflame Tech Company.
 * @par     History:
 */


# include <iostream>
# include <stdlib.h>
# include <cuda_runtime.h>
# include "cublas_v2.h"
# include <string.h>

int main (int argc, char **argv) {
  // reading cmd line arguments
  int lenA, lenB, scalar_const;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  for (int i = 1; i < 4; i++) {
    int len = sizeof(argv[i]);
    if (!strcmp(substr(argv[i], 1, 4), "lenA"))
      lenA = atoi(argv[i] + 5);
    else if (!strcmp(substr(argv[i], 1, 4), "lenB"))
      lenB = atoi(argv[i] + 5);
    else if (!strcmp(substr(argv[i], 1, 9), "const_val"))
      scalar_const = atoi(argv[i] + 10);
  }
  
  // length of vectorA and vectorB should be same
  if(lenA != lenB) {
      return EXIT_FAILURE;
  }
  
  // creating cublas handle
  cudaError_t cudaStat ;
  cublasStatus_t stat ;
  cublasHandle_t handle ;
  stat = cublasCreate(& handle);

  // allocating memory for vectors on host
  float *vectorA;
  float *vectorB;
  vectorA = (float *) malloc(lenA * sizeof (*vectorA));
  vectorB = (float *) malloc(lenB * sizeof (*vectorB));

  // setting up values in vectors
  for (int j = 0; j < lenA; j++) {
    vectorA[j] = (float)j;
  }
  for (int j = 0; j < lenB; j++) {
    vectorB[j] = (float)j;
  }

  printf ("Original vector x:\n");
  for (int j = 0; j < lenA; j++) {
    printf("%2.0f, ", vectorA[j]);
  }
  printf ("\n");
  for (int j = 0; j < lenB; j++) {
    printf ("%2.0f, ", vectorB[j]);
  }
  printf ("\n");

  // using cudamalloc for allocating memory on device
  float * vectorAA;
  float * vectorBB;
  cudaStat = cudaMalloc(( void **)& vectorAA, lenA * sizeof (*vectorA));
  cudaStat = cudaMalloc(( void **)& vectorBB, lenB * sizeof (*vectorB));

  // setting values of matrices on device
  stat = cublasSetVector(lenA, sizeof (*vectorA), vectorA, 1, vectorAA, 1);
  stat = cublasSetVector(lenB, sizeof (*vectorB), vectorB, 1, vectorBB, 1);

  // performing saxpy operation
  stat = cublasSaxpy(handle, lenA, &scalar_const, vectorAA, 1, vectorBB, 1);

  // getting the final output
  stat = cublasGetVector(lenB, sizeof(float), vectorBB, 1, vectorB, 1);

  // final output
  printf ("Final output y after Saxpy operation:\n");
  for (int j = 0; j < lenB; j++) {
    printf ("%2.0f, ", vectorB[j]);
  }
  printf ("\n");

  // free device memory
  cudaFree(vectorAA);
  cudaFree(vectorBB);

  // destroying cublas handle
  cublasDestroy(handle);

  // freeing host memory
  free(vectorA);
  free(vectorB);

  return EXIT_SUCCESS ;
}
// x,y:
// 0 , 1 , 2 , 3 , 4 , 5 ,
// y after Saxpy :
// 0 , 3 , 6 , 9 ,12 ,15 ,// a*x+y = 2*{0 ,1 ,2 ,3 ,4 ,5} + {0 ,1 ,2 ,3 ,4 ,5}
