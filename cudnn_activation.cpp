#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <time.h>

/**
 * Minimal example to apply activation on a tensor 
 * using cuDNN.
 **/
int main(int argc, char** argv)
{    
    int n, c, h, w;
    std::string a;
    
    // CAI_AG - Reading values for input parameters using command line arguments 
    for (int i = 0;i <= 5; i++)
        std::cout << argv[i] << std::endl;

    for (int i = 1; i <= 5; i++) {
        int len = sizeof(argv[i]);
        if (argv[i][1] == 'n')
          n = atoi(argv[i] + 2);
        else if (argv[i][1] == 'c')
          c = atoi(argv[i] + 2);
        else if (argv[i][1] == 'h')
          h = atoi(argv[i] + 2);
        else if (argv[i][1] == 'w')
          w = atoi(argv[i] + 2);
        else if (argv[i][1] == 'a')
          a = argv[i] + 2; 
   }

    // CAI_AG - Generating random input_data 
    int size = n*c*h*w;
    int input_data[size];
    for (int i = 0; i < size; i++)
      input_data[i] = rand() % 256;
 
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    std::cout << "Found " << numGPUs << " GPUs." << std::endl;
    
    cudaSetDevice(0); // use GPU0
    int device; 
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    std::cout << "Compute capability:" << devProp.major << "." << devProp.minor << std::endl;

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    std::cout << "Created cuDNN handle" << std::endl;

    // create the tensor descriptor
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);
    cudnnTensorDescriptor_t y_desc;
    cudnnCreateTensorDescriptor(&y_desc);
    cudnnSetTensor4dDescriptor(y_desc, format, dtype, n, c, h, w);
    
    // create the tensor
    float *x, *y;
    cudaMallocManaged(&x, size * sizeof(float));
    cudaMallocManaged(&y, size * sizeof(float));
    for(int i=0;i<size;i++) x[i] = input_data[i] * 1.00f;
    std::cout << "Original array: "; 
    for(int i=0;i<size;i++) std::cout << x[i] << " ";

    // create activation function descriptor
    float alpha[c] = {1};
    float beta[c] = {0.0};
    cudnnActivationDescriptor_t activation;
    cudnnActivationMode_t mode;
 
    // CAI_AG: Initializing activation mode 
    if (a == "tanh")
      mode = CUDNN_ACTIVATION_TANH;
    else if (a == "sigmoid")
      mode = CUDNN_ACTIVATION_SIGMOID;
    else if (a == "relu")
      mode = CUDNN_ACTIVATION_RELU;
 
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&activation);
    cudnnSetActivationDescriptor(activation, mode, prop, 0.0f);

    clock_t start, stop;
    start=clock();
    
    // Calling CuDNN Activation Forward API
    cudnnActivationForward(
        handle_,
        activation,
        alpha,
        x_desc,
        x,
        beta,
        x_desc,
        x
    );
    
    stop=clock();
	double flopsCoef = 2.0;
	for(int i=0;i<5;i++)
	std::cout <<"Input n*c*h*w........"<<size*(i+1)<< "...................latancy is "<< ((double)(stop-start))/CLOCKS_PER_SEC<< "...................Throughput "<<(i+1)* (1e-9*flopsCoef*size)/(stop-start)<<"\n";
    
    std::cout << "New array: ";
    for(int i=0;i<size;i++) std::cout << x[i] << " ";
    std::cout << std::endl;
    
    cudaFree(x);
    cudaFree(y);
    cudnnDestroy(handle_);
    std::cout << std::endl << "Destroyed cuDNN handle." << std::endl;
    
    return 0;
}
