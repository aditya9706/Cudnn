#include <iostream>
#include <string.h>
#include "cublas.h"
#include "cublas_v2.h"

#define RANDOM (rand() % 10000 * 1.00) / 100     // to generate random values

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 

int main ( int argc, char **argv) {
  // initializing size of vector with command line arguement
  cudaError_t cudaStatus ; 
  cublasStatus_t status ; 
  cublasHandle_t handle ;

  clock_t clk_start, clk_end;
  int vector_length;
  
  std::cout << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if(loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;
    
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]);
    if (!(cmd_argument.compare("-vector_length")))
      vector_length = atoi(argv[loop_count + 1]);
  }
  
  // pointers A and B pointing  to vectors
  float * HostVecA;             
  float * HostVecB; 
  
  // host memory allocation for vectors
  HostVecA = new float[vector_length]; 
  HostVecB = new float[vector_length]; 
  
  if (HostVecA == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector A)\n");
    return EXIT_FAILURE;
  }
   
  if (HostVecB == 0) {
    fprintf (stderr, "!!!! Host memory allocation error (vector B)\n");
    return EXIT_FAILURE;
  }
  
  int index; 

  // setting up values in A and B vectors
  // using RANDOM macro to generate random float numbers between 0 - 100
  for (index = 0; index < vector_length; index++) {
    HostVecA[index] = RANDOM;                               
  }

  for (index = 0; index < vector_length; index++) {
    HostVecB[index] = RANDOM; 
  }
  
  // printing the initial values in vector A and vector B
  std::cout <<"\nX vector initially:\n";
  for (index = 0; index < vector_length; index++) {
    std::cout << HostVecA[index] << " "; 
  }
  std::cout <<"\n";
  
  std::cout << "\nY vector initially :\n";
  for (index = 0; index < vector_length; index++) {
    std::cout << HostVecB[index] << " "; 
  }
  std::cout <<"\n";
  
  // Pointers for device memory allocation
  float *DeviceVecA; 
  float *DeviceVecB; 
  
  cudaStatus = cudaMalloc ((void **) &DeviceVecA, vector_length * sizeof (*HostVecA));
  if( cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A\n";
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaMalloc ((void **) &DeviceVecB, vector_length * sizeof (*HostVecB));
  if( cudaStatus != cudaSuccess) {
    std::cout <<" The device memory allocation failed for B\n";
    return EXIT_FAILURE;   
  }
 
  //initializing cublas library and setting up values for vectors in device memory same values as that present in host vectors 
  status = cublasCreate (&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }

  status = cublasSetVector (vector_length, sizeof (*HostVecA) , HostVecA, 1, DeviceVecA, 1); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set vector values for A on gpu\n");
    return EXIT_FAILURE;
  }
  
  status = cublasSetVector (vector_length, sizeof (*HostVecB), HostVecB, 1, DeviceVecB, 1); 
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to set vector values for B on gpu\n");
    return EXIT_FAILURE;
  }

  float dot_product ;

  clk_start = clock();

  // performing dot product operation and storing result in result variable
  status = cublasSdot(handle, vector_length, DeviceVecA, 1, DeviceVecB, 1, &dot_product);

  clk_end = clock();
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }
  
  //printing the final result
  std::cout << "\nDot product A.B is : " << dot_product << "\n"; 
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";

  //freeing device memory
  cudaStatus = cudaFree (DeviceVecA);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Memory free error on device for vector A\n";
    return EXIT_FAILURE;
  }
  
  cudaStatus = cudaFree (DeviceVecB);
  if( cudaStatus != cudaSuccess) {
    std::cout << " Memory free error on device for vector B\n";
    return EXIT_FAILURE;
  }
  
  //destroying cublas context and freeing host memory
  status = cublasDestroy (handle);  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to uninitialize handle");
    return EXIT_FAILURE;
  }
  
  delete[] HostVecA; 
  delete[] HostVecB; 

  return EXIT_SUCCESS ;
}

// x,y: 0, 1, 2, 3, 4, 5
// dot product x.y: = 55 
// 1*1 + 2*2 + 3*3 + 4*4 + 5*5
