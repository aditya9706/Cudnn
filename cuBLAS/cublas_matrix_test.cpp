#include <stdlib.h>
#include <stdio.h>
#include "cublas.h"
#include <iostream>
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
}

int  main(int argc, char** argv) {

  cublasStatus status;
  int i, j;
  clock_t start, end;

  // initializing cublas library
  cublasInit();

  // Reading dimensions of matrices
  for (int i = 0;i < argc; i++) {
    std::cout << argv[i] << std::endl;
  }
  int HA = atoi(argv[1]);
  int WA = atoi(argv[2]);
  int WB = atoi(argv[3]);
  int HB = WA;
  int WC = WB;
  int HC = HA;

  // allocating memory for matrices on host
  float *A = (float*) malloc(HA * WA * sizeof(float));
  float *B = (float*) malloc(HB * WB * sizeof(float));
  float *C = (float*) malloc(HC * WC * sizeof(float));

  if (A == 0) {
    fprintf (stderr, "!!!! host memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  if (B == 0) {
    fprintf (stderr, "!!!! host memory allocation error (A)\n");
      
    return EXIT_FAILURE;
  }
  if (C == 0) {
    fprintf (stderr, "!!!! host memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  // setting up values for matrices
  for (i = 0; i < HA; i++) {
    for (j = 0; j < WA; j++) {
      A[index(i, j, HA)] = (float) index(i, j, HA);
    }
  }
  for (i = 0; i < HB; i++) {
    for (j = 0; j < WB; j++) {
      B[index(i, j, HB)] = (float) index(i, j, HB);
    }
  }

  // allocating memory for matrices on device using cublasAlloc
  float* AA;
  float* BB;
  float* CC;
  status = cublasAlloc(HA*WA, sizeof(float), (void**)&AA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(HB*WB, sizeof(float), (void**)&BB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(HC*WC, sizeof(float), (void**)&CC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  // setting the values of marices on device
  status = cublasSetMatrix(HA, WA, sizeof(float), A, HA, AA, HA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix(HB, WB, sizeof(float), B, HB, BB, HB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }

  // start variable to store time
  start=clock();

  // performing matrix multiplication
  cublasSgemm('n', 'n', HA, WB, WA, 1, AA, HA, BB, HB, 0, CC, HC);

  // end variable to store time
  end=clock();

  status = cublasGetError();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error.\n");
    return EXIT_FAILURE;
  }

  // storing the final result from device matrix to host matrix
  cublasGetMatrix(HC, WC, sizeof(float), CC, HC, C, HC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device read error (A)\n");
    return EXIT_FAILURE;
  }

  // Matrix output
  printf("\nMatriz A:\n");
  printMat(A, WA, HA);
  printf("\nMatriz B:\n");
  printMat(B, WB, HB);
  printf("\nMatriz C:\n");
  printMat(C, WC, HC);

  // printing latency of the function
  double time_taken = double(end - start) / double(CLOCKS_PER_SEC);
  printf("The latency founded was  : %f\n", time_taken);

  // freeing host memory
  free(A);
  free(B);
  free(C);

  // freeing device memory
  status = cublasFree(AA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (A)\n");
    return EXIT_FAILURE;
  }

  status = cublasFree(BB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (B)\n");
    return EXIT_FAILURE;
  }

  status = cublasFree(CC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! memory free error (C)\n");
    return EXIT_FAILURE;
  }

  /* Shutdown */
  status = cublasShutdown();
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! shutdown error (A)\n");
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
