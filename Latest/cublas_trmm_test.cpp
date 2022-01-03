#include <iostream>
#include <string.h>
#include "cublas.h"
#include "cublas_v2.h"

#define FIRST_ARG "x_row"    // name for first command line argument
#define SECOND_ARG "y_col"   // name for second command line argument
#define THIRD_ARG "alpha"    // name for third command line argument
#define FIRST_ARG_LEN 5      // length of first command line argument
#define SECOND_ARG_LEN 5     // length of second command line argument
#define THIRD_ARG_LEN 5      // length of third command line argument
#define BEGIN 1
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
  int x_row, x_col, y_row, y_col, z_row, z_col;
  float alpha;

  std::cout << "\n" << std::endl;
  for (int loop_count = 0; loop_count < argc; loop_count++) {
    std::cout << argv[loop_count] << std::endl;
  }
  // reading command line arguments
  for (int loop_count = 1; loop_count < argc; loop_count++) {
    std::string str(argv[loop_count]);
    if (!((str.substr(BEGIN, FIRST_ARG_LEN)).compare(FIRST_ARG))) {
      x_row = atoi(argv[loop_count] + FIRST_ARG_LEN + 1); 
      y_row = x_row;
      z_row = x_row;
      x_col = x_row;
    }
    else if (!((str.substr(BEGIN, SECOND_ARG_LEN)).compare(SECOND_ARG))) {
      y_col = atoi(argv[loop_count] + SECOND_ARG_LEN + 1);
      z_col = y_col;
    }
    else if (!((str.substr(BEGIN, THIRD_ARG_LEN)).compare(THIRD_ARG))) {
      alpha = atoi(argv[loop_count] + THIRD_ARG_LEN + 1);
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
  float *HostMatX = new float[x_row * x_col];
  float *HostMatY = new float[y_row * y_col];
  float *HostMatZ = new float[z_row * z_col];

  if (HostMatX == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixX)\n");
    return EXIT_FAILURE;
  }
  if (HostMatY == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixY)\n");
    return EXIT_FAILURE;
  }
  if (HostMatZ == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrixZ)\n");
    return EXIT_FAILURE;
  }

  // setting up values for matrices
  // using RANDOM macro to generate random numbers between 0 - 100
  int row, col;
  for (col = 0; col < x_col; col++) {
    for (row = 0; row < x_row; row++) {
      if (row >= col) 
        HostMatX[INDEX(row, col, x_row)] = RANDOM;
      else 
        HostMatX[INDEX(row, col, x_row)] = 0.0;
    }
  }

  for (row = 0; row < y_row; row++) {
    for (col = 0; col < y_col; col++) {
        HostMatY[INDEX(row, col, y_row)] = RANDOM;
    }
  }

  // allocating memory for matrices on device using cublasAlloc
  float* DeviceMatX;
  float* DeviceMatY;
  float* DeviceMatZ;

  status = cublasAlloc(x_row * x_col, sizeof(float), (void**) &DeviceMatX);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (X)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(y_row * y_col, sizeof(float), (void**) &DeviceMatY);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Device memory allocation error (Y)\n");
    return EXIT_FAILURE;
  }
  status = cublasAlloc(z_row * z_col, sizeof(float), (void**) &DeviceMatZ);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! device memory allocation error (Z)\n");
    return EXIT_FAILURE;
  }

  // setting the values of matrices on device
  status = cublasSetMatrix(x_row, x_col, sizeof(float), HostMatX, x_row, DeviceMatX, x_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (X) failed\n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix(y_row, y_col, sizeof(float), HostMatY, y_row, DeviceMatY, y_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (X) failed\n");
    return EXIT_FAILURE;
  }
  
  // start variable to store time
  clk_start = clock();
  
  // triangular matrix - matrix multiplication : d_z = alpha * d_x * d_y ;
  // d_x - mxm triangular matrix in lower mode ,
  // d_y , d_z - mxn general matrices ; alpha - scalar
  status = cublasStrmm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                       CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, y_row, y_col, &alpha, 
                       DeviceMatX, x_row, DeviceMatY, y_row, DeviceMatZ, z_row);
 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Setting up values on device for matrix (Y) failed\n");
    return EXIT_FAILURE;
  }

  // end variable to store time
  clk_end = clock();

  // getting the final output
  status = cublasGetMatrix(z_row, z_col, sizeof(float), DeviceMatZ, z_row, HostMatZ, z_row);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to Get values in Host vector Y\n");
    return EXIT_FAILURE;
  }

  // Matrix output
  std::cout << "\nMatrix X:";
  PrintMatrix(HostMatX, x_row, x_col);
  std::cout << "\nMatrix Y:";
  PrintMatrix(HostMatY, y_row, y_col);
  std::cout << "\nMatrix Z:";
  PrintMatrix(HostMatZ, z_row, z_col);

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";

  // free device memory
  cudaStatus = cudaFree(DeviceMatX);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory deallocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(DeviceMatY);
  if( cudaStatus != cudaSuccess) {
    printf(" the device  memory deallocation failed\n");
    return EXIT_FAILURE;   
  }
  cudaStatus = cudaFree(DeviceMatZ);
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
  delete[] HostMatX; // free host memory
  delete[] HostMatY; // free host memory
  delete[] HostMatZ; // free host memory
  
  return EXIT_SUCCESS ;
}