#include <iostream>
#include <string.h>
#include "cublas.h"
#include "cublas_v2.h"

#define INDEX(row, col, row_count) (((col) * (row_count)) + (row))     // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100    // to generate random values 

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

void PrintMatrix(float* Matrix, int batch_count, int matrix_row, int matrix_col) {
  int row, col, batch;
  for (batch = 0; batch < batch_count; batch++) {
    std::cout << "\nBatch " << batch << ": \n";
    for (row = 0; row < matrix_row; row++) {
      for (col = 0; col < matrix_col; col++) {
        std::cout << Matrix[INDEX(row, col, matrix_row) + batch * matrix_row * matrix_col] << " ";
      }
      std::cout << "\n";
    }
  }
  std::cout << "\n";
}

int main (int argc, char **argv) {
  clock_t clk_start, clk_end;
  int A_row, A_col, B_row, B_col, C_row, C_col, batch_count;
  float alpha, beta;

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
    }

    else if (!(cmd_argument.compare("-A_column"))) {
      A_col = atoi(argv[loop_count + 1]); 
    }

    else if (!(cmd_argument.compare("-B_column"))) {
      B_col = atoi(argv[loop_count + 1]);
    }

    else if (!(cmd_argument.compare("-batch_count"))) {
      batch_count = atoi(argv[loop_count + 1]);
    }
    
    else if (!(cmd_argument.compare("-alpha"))) {
      alpha = atof(argv[loop_count + 1]);
    }

    else if (!(cmd_argument.compare("-beta"))) {
      beta = atof(argv[loop_count + 1]);
    }
  }

  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
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
  float *HostMatA = new float[batch_count * A_row * A_col];
  float *HostMatB = new float[batch_count * B_row * B_col];
  float *HostMatC = new float[batch_count * C_row * C_col];

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
  int row, col, batch;

  for (batch = 0; batch < batch_count; batch++) {
    for (col = 0; col < A_col; col++) {
      for (row = 0; row < A_row; row++) {
          HostMatA[INDEX(row, col, A_row) + batch * A_col * A_row] = RANDOM;
      }
    }
  }

  for (batch = 0; batch < batch_count; batch++) {
    for (row = 0; row < B_row; row++) {
      for (col = 0; col < B_col; col++) {
          HostMatB[INDEX(row, col, B_row) + batch * B_col * B_row] = RANDOM;
      }
    }
  }

  for (batch = 0; batch < batch_count; batch++) {
    for (row = 0; row < C_row; row++) {
      for (col = 0; col < C_col; col++) {
          HostMatC[INDEX(row, col, C_row) + batch * C_col * C_row] = RANDOM;
      }
    }
  }

  std::cout << "\nMatrix A:\n";
  PrintMatrix(HostMatA, batch_count, A_row, A_col);
  std::cout << "\nMatrix B:\n";
  PrintMatrix(HostMatB, batch_count, B_row, B_col);
  std::cout << "\nMatrix C:\n";
  PrintMatrix(HostMatC, batch_count, C_row, C_col);

  // allocating memory for matrices on device using cublasAlloc
  float* DeviceMatA;
  float* DeviceMatB;
  float* DeviceMatC;

  status = cublasAlloc(batch_count * A_row * A_col, sizeof(float), (void**) &DeviceMatA);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (A)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(batch_count * B_row * B_col, sizeof(float), (void**) &DeviceMatB);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (B)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(batch_count * C_row * C_col, sizeof(float), (void**) &DeviceMatC);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  // setting the values of matrices on device
  cudaStatus = cudaMemcpy(DeviceMatA, HostMatA, sizeof(float) * batch_count * A_row * A_col, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (A) failed\n");
    return EXIT_FAILURE;
  }
  cudaStatus = cudaMemcpy(DeviceMatB, HostMatB, sizeof(float) * batch_count * B_row * B_col, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (B) failed\n");
    return EXIT_FAILURE;
  }
  cudaStatus = cudaMemcpy(DeviceMatC, HostMatC, sizeof(float) * batch_count * C_row * C_col, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (C) failed\n");
    return EXIT_FAILURE;
  }
  
  // defining stride to differentiate between each batch
  long long int strideA = A_row * A_col;
  long long int strideB = B_row * B_col;
  long long int strideC = C_row * C_col;
  
  // start variable to store time
  clk_start = clock();
  
  // solve d_A * X = alpha * d_B
  // the solution X overwrites rhs d_B
  // d_A - m x m triangular matrix in lower mode
  // d_B, X - m x n general matrices
  // alpha - scalar
  status = cublasSgemmStridedBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                     A_row, B_col, A_col, &alpha, DeviceMatA,
                                     A_row, strideA, DeviceMatB, B_row, strideB,
                                     &beta, DeviceMatC, C_row, strideC, batch_count);

  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! API execution failed\n");
    return EXIT_FAILURE;
  }

  // end variable to store time
  clk_end = clock();

  // getting the final output
  cudaStatus = cudaMemcpy(HostMatC, DeviceMatC,  sizeof(float) * batch_count * C_row * C_col, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    fprintf (stderr, "!!!! Failed to to Get values in Host Matrix C");
    return EXIT_FAILURE;
  }

  // Matrix output
  std::cout << "\nMatrix C after gemmStridedBatched operation:\n";
  PrintMatrix(HostMatC, batch_count, C_row, C_col);  

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

  delete[] HostMatA; // free host memory
  delete[] HostMatB; // free host memory
  delete[] HostMatC; // free host memory

  return EXIT_SUCCESS ;
}
