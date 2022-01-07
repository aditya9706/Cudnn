#include <iostream>
#include <string.h>
#include "cublas.h"
#include "cublas_v2.h"

#define INDEX(row, col, row_count) (((col)*(row_count))+(row))   // for getting index values matrices
#define RANDOM (rand() % 10000 * 1.00) / 100      // for getting random values

/* 1e-9 for converting throughput in GFLOP/sec, multiplying by 2 as each multiply-add operation uses two flops and 
 finally dividing it by latency to get required throughput */
#define THROUGHPUT(clk_start, clk_end)  ((1e-9 * 2) / (clk_end - clk_start)) 


void PrintMatrix(cuComplex* Matrix, int matrix_row, int matrix_col) {
  int row, col;
  for (row = 0; row < matrix_row; row++) {
    for (col = 0; col < matrix_col; col++) {
      std::cout << Matrix[INDEX(row, col, matrix_row)].x << "+" 
                << Matrix[INDEX(row, col, matrix_row)].y << "*I ";
    }
    std::cout << "\n";
  }
}

int main (int argc, char **argv) {
  int A_row, A_col, B_row, B_col, C_row, C_col;
  float alpha_real, alpha_imaginary, beta_real, beta_imaginary;

  std::cout << "\n\n" << argv[0] << std::endl;
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::cout << argv[loop_count] << " ";
    if(loop_count + 1 < argc)
      std::cout << argv[loop_count + 1] << std::endl;
  }
  std::cout << std::endl;
  
  // reading cmd line arguments
  for (int loop_count = 1; loop_count < argc; loop_count += 2) {
    std::string cmd_argument(argv[loop_count]); 

    if (!(cmd_argument.compare("-A_row")))
      A_row = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-B_column")))
      B_col = atoi(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_real")))
      alpha_real = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-alpha_imaginary")))
      alpha_imaginary = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_real")))
      beta_real = atof(argv[loop_count + 1]);

    else if (!(cmd_argument.compare("-beta_imaginary")))
      beta_imaginary = atof(argv[loop_count + 1]);
  }
  
  A_col = A_row;
  B_row = A_col;
  C_row = A_row;
  C_col = B_col;
  
  cudaError_t cudaStatus;
  cublasStatus_t status;
  cublasHandle_t handle;

  time_t clk_start, clk_end;

  cuComplex *HostMatA;
  cuComplex *HostMatB;
  cuComplex *HostMatC;

  HostMatA = new cuComplex[A_row * A_col]; // host memory alloc for A
  HostMatB = new cuComplex[B_row * B_col]; // host memory alloc for B
  HostMatC = new cuComplex[C_row * C_col]; // host memory alloc for C
  
  if (HostMatA == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix A)\n");
    return EXIT_FAILURE;
  }
  if (HostMatB == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix B)\n");
    return EXIT_FAILURE;
  }
  if (HostMatC == 0) {
    fprintf (stderr, "!!!! host memory allocation error (matrix C)\n");
    return EXIT_FAILURE;
  }
  
  int row, col;
  // define the lower triangle of an mxm Hermitian matrix A in lower mode column by column
  for (col = 0; col < A_col; col++) {                 
    for (row = 0; row < A_row; row++) {                                   
      if(row >= col) {                                        
        HostMatA[INDEX(row, col, A_row)].x = RANDOM;                   
        HostMatA[INDEX(row, col, A_row)].y = 0.0f;                       
      }                                                           
    }
  }

  // print the lower triangle of A row by row
  std::cout << "\nLower triangle of Matrix A :\n";
  for (row = 0; row < A_row; row++){
    for (col = 0; col < A_col; col++) {
      if(row >= col) {
        std::cout << HostMatA[INDEX(row, col, A_row)].x << "+" << HostMatA[INDEX(row, col, A_row)].y << "*I ";                              
      }
    }
  std::cout << "\n";
  }

  // define mxn matrix B column by column
  for(col = 0; col < B_col; col++) {           
    for(row = 0; row < B_row; row++) {                      
      HostMatB[INDEX(row, col, B_row)].x = RANDOM;            
      HostMatB[INDEX(row, col, B_row)].y = 0.0f;                   
                   
    }
  }
  
  // define mxn matrix C column by column 
  for(col = 0; col < C_col; col++) {           
    for(row = 0; row < C_row; row++) {                      
      HostMatC[INDEX(row, col, C_row)].x = RANDOM;              
      HostMatC[INDEX(row, col, C_row)].y = 0.0f;                 
    }
  }
  
  // print B row by row
  std::cout << "\nMatrix B:\n";
  PrintMatrix(HostMatB, B_row, B_col);
  
  // print C row by row
  std::cout << "\nMatrix C:\n";
  PrintMatrix(HostMatC, C_row, C_col);
 
  // on the device
  cuComplex* DeviceMatA; // d_A - A on the device
  cuComplex* DeviceMatB; // d_B - B on the device
  cuComplex* DeviceMatC; // d_C - C on the device

  // device memory alloc for A
  cudaStatus = cudaMalloc ((void **)&DeviceMatA , A_row * A_col * sizeof(cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for A\n";
    return EXIT_FAILURE;
  }
  
  // device memory alloc for B
  cudaStatus = cudaMalloc ((void **)&DeviceMatB , B_row * B_col * sizeof(cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for B\n";
    return EXIT_FAILURE;
  }
  
  // device memory alloc for C
  cudaStatus = cudaMalloc ((void **)&DeviceMatC, C_row * C_col * sizeof(cuComplex));
  if(cudaStatus != cudaSuccess) {
    std::cout << " The device memory allocation failed for C\n";
    return EXIT_FAILURE;
  }
  
  // initialize CUBLAS context
  status = cublasCreate (& handle);  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Failed to initialize handle\n");
    return EXIT_FAILURE;
  }
  
  // copy matrices from the host to the device
  status = cublasSetMatrix (A_row, A_col, sizeof (*HostMatA) , HostMatA, A_row, DeviceMatA, A_row); //A -> d_A
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix A from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (B_row, B_col, sizeof (*HostMatB) , HostMatB, B_row, DeviceMatB, B_row); //B -> d_B
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix B from host to device failed \n");
    return EXIT_FAILURE;
  }
  status = cublasSetMatrix (C_row, C_col, sizeof (*HostMatC) , HostMatC, C_row, DeviceMatC, C_row); //C -> d_C
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from host to device failed \n");
    return EXIT_FAILURE;
  }

  // defining alpha and beta
  cuComplex alpha = {alpha_real, alpha_imaginary};
  cuComplex beta = {beta_real, beta_imaginary};
  
  clk_start = clock();
  
  // Hermitian matrix - matrix multiplication :
  // d_C = alpha * d_A * d_B + beta * d_C;
  // d_A - m x m hermitian matrix ; d_B, d_C - m x n - general matrices ;
  // alpha, beta - scalars
  status = cublasChemm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
                       A_row, B_col, &alpha, DeviceMatA, A_row, DeviceMatB,
                       B_row, &beta, DeviceMatC, C_row);
  
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! kernel execution error\n");
    return EXIT_FAILURE;
  }

  clk_end = clock();
  
  status = cublasGetMatrix(C_row, C_col, sizeof(*HostMatC), DeviceMatC, C_row, HostMatC, C_row); // d_C -> C
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "Copying matrix C from device to host failed\n");
    return EXIT_FAILURE;
  }

  printf ("\nMatrix C after Chemm :\n");
  PrintMatrix(HostMatC, C_row, C_col);
  
  
  // printing latency and throughput of the function
  std::cout << "\nLatency: " <<  ((double)(clk_end - clk_start)) / double(CLOCKS_PER_SEC) <<
               "\nThroughput: " << THROUGHPUT(clk_start, clk_end) << "\n\n";
  
  //free device memory
  cudaStatus = cudaFree(DeviceMatA); 
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for A\n";
    return EXIT_FAILURE;   
  }
  // free device memory
  cudaStatus = cudaFree(DeviceMatB); 
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for B\n";
    return EXIT_FAILURE;   
  }
  // free device memory
  cudaStatus = cudaFree(DeviceMatC);
  if( cudaStatus != cudaSuccess) {
    std::cout << " the device memory deallocation failed for C\n";
    return EXIT_FAILURE;   
  }
  
  status  = cublasDestroy (handle); // destroy CUBLAS context
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf (stderr, "!!!! Unable to uninitialize handle \n");
    return EXIT_FAILURE;
  } 
  
  delete[] HostMatA; // free host memory
  delete[] HostMatB; // free host memory
  delete[] HostMatC; // free host memory
  
  return EXIT_SUCCESS;
}

// lower triangle of a:
// 11+ 0*I
// 12+ 0*I 17+ 0*I
// 13+ 0*I 18+ 0*I 22+ 0*I
// 14+ 0*I 19+ 0*I 23+ 0*I 26+ 0*I
// 15+ 0*I 20+ 0*I 24+ 0*I 27+ 0*I 29+ 0*I
// 16+ 0*I 21+ 0*I 25+ 0*I 28+ 0*I 30+ 0*I 31+ 0*I
// b, c:
// 11+ 0*I 17+ 0*I 23+ 0*I 29+ 0*I 35+ 0*I
// 12+ 0*I 18+ 0*I 24+ 0*I 30+ 0*I 36+ 0*I
// 13+ 0*I 19+ 0*I 25+ 0*I 31+ 0*I 37+ 0*I
// 14+ 0*I 20+ 0*I 26+ 0*I 32+ 0*I 38+ 0*I
// 15+ 0*I 21+ 0*I 27+ 0*I 33+ 0*I 39+ 0*I
// 16+ 0*I 22+ 0*I 28+ 0*I 34+ 0*I 40+ 0*I

// c after Chemm :
// 1122+0* I 1614+0* I 2106+0* I 2598+0* I 3090+0* I //
// 1484+0* I 2132+0* I 2780+0* I 3428+0* I 4076+0* I //
// 1740+0* I 2496+0* I 3252+0* I 4008+0* I 4764+0* I // c = alpha * a * b + beta * c
// 1912+0* I 2740+0* I 3568+0* I 4396+0* I 5224+0* I //
// 2025+0* I 2901+0* I 3777+0* I 4653+0* I 5529+0* I //
// 2107+0* I 3019+0* I 3931+0* I 4843+0* I 5755+0* I //
