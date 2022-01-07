#include <iostream>
#include <string.h>
#include "cublas.h"
#include "cublas_v2.h"

#define RANDOM (rand() % 10000 * 1.00) / 100    // for getting random values

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

void PrintVector(float* Vector, int length) {
  for (int index = 0; index < length; index++) {
    std::cout << Vector[index] << " ";
  }
  std::cout << std::endl;
}

int main (int argc, char **argv) {
  clock_t clk_start, clk_end;
  int vector_length;
  float alpha;

  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if(loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;

  // for reading command line arguements
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);  
    
    if (!(cmd_argument.compare("-vector_length")))
      vector_length = atoi(argv[loop_count + 1]);
    
    else if (!(cmd_argument.compare("-alpha")))
      alpha = atoi(argv[loop_count + 1]);
  }

  cudaError_t cudaStatus ;
  cublasStatus_t status ;
  cublasHandle_t handle ;

  // creating cublas handle
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  // allocating memory for vectors on host
  float *HostVecX;
  float *HostVecY;  
  HostVecX = new float[vector_length]; 
  HostVecY = new float[vector_length];

  if (HostVecX == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector X)\n");
    return EXIT_FAILURE;
  }
   
  if (HostVecY == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector Y)\n");
    return EXIT_FAILURE;
  }

  // setting up values in vectors
  for (int index = 0; index < vector_length; index++) {
    HostVecX[index] = RANDOM;
  }
  for (int index = 0; index < vector_length; index++) {
    HostVecY[index] = RANDOM;
  }

  std::cout << "\nOriginal vector X:\n";
  PrintVector(HostVecX, vector_length);


  std::cout << "\nOriginal vector Y:\n";
  PrintVector(HostVecY, vector_length);

  // using cudamalloc for allocating memory on device
  float * DeviceVecX;
  float * DeviceVecY;

  cudaStatus = cudaMalloc((void **)&DeviceVecX, vector_length * sizeof(*HostVecX));
  if( cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed\n");
    return EXIT_FAILURE;
  }

  cudaStatus = cudaMalloc((void **)&DeviceVecY, vector_length * sizeof(*HostVecY));
  if( cudaStatus != cudaSuccess) {
    printf(" The device memory allocation failed\n");
    return EXIT_FAILURE;
  }

  // setting values of matrices on device
  status = cublasSetVector(vector_length, sizeof(*HostVecX), HostVecX, 1, DeviceVecX, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set up values in device vector X\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector(vector_length, sizeof(*HostVecY), HostVecY, 1, DeviceVecY, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to set up values in device vector Y\n");
    return EXIT_FAILURE;
  }

  clk_start = clock();

  // performing axpy operation
  // Y = alpha * X + Y
  status = cublasSaxpy(handle, vector_length, &alpha, DeviceVecX, 1, DeviceVecY, 1);
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }

  clk_end = clock();

  // getting the final output
  status = cublasGetVector(vector_length, sizeof(float), DeviceVecY, 1, HostVecY, 1);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to to Get values in Host vector Y\n");
    return EXIT_FAILURE;
  }

  // final output
  std::cout << "\nFinal output Y after axpy operation:\n";
  PrintVector(HostVecY, vector_length);

  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end);

  // free device memory
  cudaStatus = cudaFree(DeviceVecX);
  if( cudaStatus != cudaSuccess) {
    printf(" the device memory deallocation failed\n");
    return EXIT_FAILURE;
  }
  cudaStatus = cudaFree(DeviceVecY);
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
  delete[] HostVecX;
  delete[] HostVecY;

  return EXIT_SUCCESS ;
}
// x, y:
// 0 , 1 , 2 , 3 , 4 , 5
// y after axpy :
// 0 , 3 , 6 , 9 ,12 ,15 ,// a * x + y = 2 * {0 ,1 ,2 ,3 ,4 ,5} + {0 ,1 ,2 ,3 ,4 ,5}
