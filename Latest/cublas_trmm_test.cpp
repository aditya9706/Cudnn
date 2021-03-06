#include <iostream>
#include <string.h>
#include "cublas.h"
#include "cublas_v2.h"

#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))     // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100    // to generate random values 

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

void PrintMatrix(float* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    std::cout << "\n";
    for (col = 0; col < matrix_col; col++) {
      std::cout << Matrix[INDEX(row, col, matrix_row)] << " ";
    }
  }
  std::cout << "\n";
}

int main (int argc, char **argv) {

  clock_t clk_start, clk_end;
  int A_row, A_col, B_row, B_col, C_row, C_col;
  float alpha;

  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if(loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  // reading command line arguments
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);

    if (!(cmd_argument.compare("-A_row"))) {
      A_row = atoi(argv[loop_count + 1]); 
      B_row = A_row;
      C_row = A_row;
      A_col = A_row;
    }
    else if (!(cmd_argument.compare("-B_column"))) {
      B_col = atoi(argv[loop_count + 1]);
      C_col = B_col;
    }
    else if (!(cmd_argument.compare("-alpha"))) {
      alpha = atoi(argv[loop_count + 1]);
    }
  }
  
  // creating cublas handle
  cudaError_t cudaStatus;
  cublasStatus_t status;
  cublasHandle_t handle;

  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  // allocating memory for matrices on host
  float *HostMatA = new float[A_row * A_col];
  float *HostMatB = new float[B_row * B_col];
  float *HostMatC = new float[C_row * C_col];

  if (HostMatA == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixA)\n");
    return EXIT_FAILURE;
  }
  if (HostMatB == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixB)\n");
    return EXIT_FAILURE;
  }
  if (HostMatC == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixC)\n");
    return EXIT_FAILURE;
  }

  // setting up values for matrices
  // using RANDOM macro to generate random numbers between 0 - 100
  int row, col;
  for (col = 0; col < A_col; col++) {
    for (row = 0; row < A_row; row++) {
      if (row >= col) 
        HostMatA[INDEX(row, col, A_row)] = RANDOM;
      else 
        HostMatA[INDEX(row, col, A_row)] = 0.0;
    }
  }

  for (row = 0; row < B_row; row++) {
    for (col = 0; col < B_col; col++) {
        HostMatB[INDEX(row, col, B_row)] = RANDOM;
    }
  }

  // allocating memory for matrices on device using cublasAlloc
  float* DeviceMatA;
  float* DeviceMatB;
  float* DeviceMatC;

  status = cublasAlloc(A_row * A_col, sizeof(float), (void**) &DeviceMatA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(B_row * B_col, sizeof(float), (void**) &DeviceMatB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (B)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(C_row * C_col, sizeof(float), (void**) &DeviceMatC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  // setting the values of matrices on device
  status = cublasSetMatrix(A_row, A_col, sizeof(float), HostMatA, A_row, DeviceMatA, A_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (A) failed\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix(B_row, B_col, sizeof(float), HostMatB, B_row, DeviceMatB, B_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (B) failed\n");
    return EXIT_FAILURE;
  }
  
  // start variable to store time
  clk_start = clock();
  
  // triangular matrix - matrix multiplication : d_C = alpha * d_A * d_B ;
  // d_A - mxm triangular matrix in lower mode ,
  // d_B , d_C - mxn general matrices ; alpha - scalar
  status = cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                       CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, B_row, B_col, &alpha, 
                       DeviceMatA, A_row, DeviceMatB, B_row, DeviceMatC, C_row);
 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! API execution failed\n");
    return EXIT_FAILURE;
  }

  // end variable to store time
  clk_end = clock();

  // getting the final output
  status = cublasGetMatrix(C_row, C_col, sizeof(float), DeviceMatC, C_row, HostMatC, C_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to Get values in Host vector C\n");
    return EXIT_FAILURE;
  }

  // Matrix output
  std::cout << "\nMatrix A:";
  PrintMatrix(HostMatA, A_row, A_col);
  std::cout << "\nMatrix B:";
  PrintMatrix(HostMatB, B_row, B_col);
  std::cout << "\nMatrix C:";
  PrintMatrix(HostMatC, C_row, C_col);

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";

  // free device memory
  cudaStatus = cudaFree(DeviceMatA);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory deallocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(DeviceMatB);
  if( cudaStatus != cudaSuccess) {
    printf(" the device  memory deallocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(DeviceMatC);
  if( cudaStatus != cudaSuccess) {
    printf(" the device  memory deallocation failed\n");
    return EXIT_FAILURE;   
  }

  // destroying cublas handle
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to uninitialize");
    return EXIT_FAILURE;
  }

  // freeing host memory
  delete[] HostMatA; // free host memory
  delete[] HostMatB; // free host memory
  delete[] HostMatC; // free host memory
  
  return EXIT_SUCCESS ;
}

// lower triangle of a:
// 11
// 12 17
// 13 18 22
// 14 19 23 26
// 15 20 24 27 29
// 16 21 25 28 30 31

// b:
// 11 17 23 29 35
// 12 18 24 30 36
// 13 19 25 31 37
// 14 20 26 32 38
// 15 21 27 33 39
// 16 22 28 34 40

// c = alpha * a * b

// c after Strmm :
// 121 187 253 319 385
// 336 510 684 858 1032
// 645 963 1281 1599 1917
// 1045 1537 2029 2521 3013
// 1530 2220 2910 3600 4290
// 2091 2997 3903 4809 5715

