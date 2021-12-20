#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"
#include <iostream>
#include<string.h>
#include <time.h>

#define index(i,j,ld) (((j)*(ld))+(i))

void printMat(float*P, int uWP, int uHP) {
  int i, j;
  for (i = 0; i < uHP; i++) {
    printf("\n");
    for (j = 0; j < uWP; j++) {
      printf("%f ", P[index(i, j, uHP)]);
    }
  }
  printf("\n\n");
}

char* substr(char* arr, int begin, int len)
{
    char* res = new char[len + 1];
    for (int i = 0; i < len; i++)
        res[i] = *(arr + begin + i);
    res[len] = 0;
    return res;
}

int  main(int argc, char** argv) {

  cublasStatus status;
  int i, j;
  clock_t start, end;

  // initializing cublas library
  cublasInit();

  // Reading dimensions of matrices
  int rowA, colA, rowB, colB, rowC, colC;
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  for (int i = 1; i < 5; i++) {
        int len = sizeof(argv[i]);
        if (!strcmp(substr(argv[i], 1, 4), "rowA"))
          rowA = atoi(argv[i] + 5);
        else if (!strcmp(substr(argv[i], 1, 4), "colA"))
          colA = atoi(argv[i] + 5);
        else if (!strcmp(substr(argv[i], 1, 4), "rowB"))
          rowB = atoi(argv[i] + 5);
        else if (!strcmp(substr(argv[i], 1, 4), "colB"))
          colB = atoi(argv[i] + 5);
  }
  rowC = rowA;
  colC = colB;
  
  // allocating memory for matrices on host
  float *matrixA = (float*) malloc(rowA * colA * sizeof(float));
  float *matrixB = (float*) malloc(rowB * colB * sizeof(float));
  float *matrixC = (float*) malloc(rowC * colC * sizeof(float));

  if (matrixA == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixA)\n");
    return EXIT_FAILURE;
  }
  if (matrixB == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixB)\n");
    return EXIT_FAILURE;
  }
  if (matrixC == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixC)\n");
    return EXIT_FAILURE;
  }

  // setting up values for matrices
  for (i = 0; i < rowA; i++) {
    for (j = 0; j < colA; j++) {
      matrixA[index(i, j, rowA)] = (rand() % 10000 * 1.00) / 100;
    }
  }
  for (i = 0; i < rowB; i++) {
    for (j = 0; j < colB; j++) {
      matrixB[index(i, j, rowB)] = (rand() % 10000 * 1.00) / 100;
    }
  }

  // allocating memory for matrices on device using cublasAlloc
  float* matrixAA;
  float* matrixBB;
  float* matrixCC;
  status = cublasAlloc(rowA * colA, sizeof(float), (void**)& matrixAA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(rowB * colB, sizeof(float), (void**)& matrixBB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(rowC * colC, sizeof(float), (void**)& matrixCC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  // setting the values of matrices on device
  status = cublasSetMatrix(rowA, colA, sizeof(float), matrixA, rowA, matrixAA, rowA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix(rowB, colB, sizeof(float), matrixB, rowB, matrixBB, rowB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  // start variable to store time
  start = clock();

  // performing matrix multiplication
  cublasSgemm('n', 'n', rowA, colB, colA, 1, matrixAA, rowA, matrixBB, rowB, 0, matrixCC, rowC);

  // end variable to store time
  end = clock();

  status = cublasGetError();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }

  // storing the final result from device matrix to host matrix
  cublasGetMatrix(rowC, colC, sizeof(float), matrixCC, rowC, matrixC, rowC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device read error (A)\n");
    return EXIT_FAILURE;
  }

  // Matrix output
  printf("\nMatriz A:\n");
  printMat(matrixA, colA, rowA);
  printf("\nMatriz B:\n");
  printMat(matrixB, colB, rowB);
  printf("\nMatriz C:\n");
  printMat(matrixC, colC, rowC);

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(end-start)) / double(CLOCKS_PER_SEC) <<
        "\nThroughput: " << (1e-9 * 2) / (end - start) << "\n\n";

  // freeing host memory
  free(matrixA);
  free(matrixB);
  free(matrixC);

  // freeing device memory
  status = cublasFree(matrixAA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (matrixA)\n");
    return EXIT_FAILURE;
  }

  status = cublasFree(matrixBB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (matrixB)\n");
    return EXIT_FAILURE;
  }

  status = cublasFree(matrixCC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (matrixC)\n");
    return EXIT_FAILURE;
  }

  /* Shutdown */
  status = cublasShutdown();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! shutdown error (matrixA)\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
