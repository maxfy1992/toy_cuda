// Includes
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <cstring>

// includes, CUDA
#include <cuda.h>
#include <builtin_types.h>

// Driver API Variables
CUdevice cuDevice;
CUcontext cuContext;
CUmodule cuModule;
CUfunction toy_kernel;

// buffer host and guest
float *h_A;
float *h_B;
float *h_C;
float *h_C_cpu;
CUdeviceptr d_A;
CUdeviceptr d_B;
CUdeviceptr d_C;

// Functions
void exit_fail();

////////////////////////////////////////////////////////////////////////////////
// These are CUDA Helper functions

// This will output the proper CUDA error strings in the event that a CUDA host call returns an error
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

// These are the inline versions for all of the SDK helper functions
inline void __checkCudaErrors(CUresult err, const char *file, const int line)
{
    if (CUDA_SUCCESS != err)
    {
        fprintf(stderr, "checkCudaErrors() Driver API error = %04d from file <%s>, line %i.\n",
                err, file, line);
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    printf("Toy cuda run(Driver API) ...\n");
    int devID = 0;
    int M = 2;//atoi(argv[1]);
    int N = 3;//atoi(argv[2]);
    int K = 4;//atoi(argv[3]);
    char *cubin_file = "toy_kernel.cubin";//argv[4];
    char *func_name = "toy_kernel";  
 
    CUresult error;
    
    // Initialize
    checkCudaErrors(cuInit(0));
    printf("--initialize success!\n");

    // Get number of devices supporting CUDA
    int deviceCount = 0;
    error = cuDeviceGetCount(&deviceCount);
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    if (deviceCount == 0)
    {
        printf("There is no device supporting CUDA.\n");
        exit_fail();
    }
    printf("--DeviceGetCount success!,get %d device(s)\n",deviceCount);
    
    int major, minor;
    char deviceName[100];
    checkCudaErrors(cuDeviceComputeCapability(&major, &minor, devID));
    checkCudaErrors(cuDeviceGetName(deviceName, 256, devID));
    printf("--Using Device %d: \"%s\" with Compute %d.%d capability\n", devID, deviceName, major, minor);

    // pick up device with zero ordinal (default, or devID)
    error = cuDeviceGet(&cuDevice, devID);
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    printf("--getDevice success\n");

   // Create context
    error = cuCtxCreate(&cuContext, 0, cuDevice);
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    printf("--create context success!\n");

    //load cubin module
    error = cuModuleLoad(&cuModule, cubin_file);
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    printf("--load cubin success!\n");

    error = cuModuleGetFunction(&toy_kernel, cuModule, func_name);
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    printf("--get function success!\n");    

    size_t size_A = M * K * sizeof(float);
    // Allocate input  h_A and h_B in host memory
    h_A = (float *)malloc(size_A);
    if (h_A == NULL)
    {
        exit_fail();;
    }
    printf("--alloc h_A success!\n");

    size_t size_B = K * N * sizeof(float);
    h_B = (float *)malloc(size_B);
    if (h_B == NULL)
    {
        exit_fail();
    }
    printf("--alloc h_B success!\n");

    size_t size_C = N * M * sizeof(float);
    h_C = (float *)malloc(size_C);
    if (h_C == NULL)
    {
        exit_fail();
    }
    printf("--alloc h_C success!\n");

    h_C_cpu = (float *)malloc(size_C);
    if (h_C_cpu == NULL)
    {
        exit_fail();
    }
    printf("--alloc h_C_cpu success!\n");

    // Allocate in device memory
    error = cuMemAlloc(&d_A, size_A);
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    printf("--alloc d_A success!\n");
    error = cuMemAlloc(&d_B, size_B); 
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    printf("--alloc d_B success!\n");

    error = cuMemAlloc(&d_C, size_C); 
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    printf("--alloc d_C success!\n");
    
    // init input
    for(int i=0;i<M*N;i++)
    {
       h_C[i] = 0;
       h_C_cpu[i] = 0;
    }

    for(int i=0;i<M*K;i++)
       h_A[i] = i;
    for(int i=0;i<N*K;i++)
       h_B[i] = 1.0;
    
    //
    for (int i = 0; i < M; ++i)
        {
            for(int j = 0; j < K; ++j)
            {
                printf("A[%d][%d]=(%f). \n", i,j, h_A[i*K+j]);
            }
        } 

    error = cuMemcpyHtoD(d_A, h_A, size_A);
    if (error != CUDA_SUCCESS)
    {
        exit_fail(); 
    }
    printf("--memcpy h_A to devivce d_A  success!\n");

    error = cuMemcpyHtoD(d_B, h_B, size_B);
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    printf("--memcpy h_B to devivce d_B  success!\n");

    error = cuMemcpyHtoD(d_C, h_C, size_C);
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    printf("--memcpy h_C to devivce d_C  success!\n");

    void *args[] = {&d_A, &d_B, &d_C, &M, &N, &K};
    
    //Launch the CUDA kernel
    checkCudaErrors(cuLaunchKernel(toy_kernel, 1, 1, 1,
                           32, 1, 1, 
                           0,
                           NULL, args, NULL));
    printf("--launch kernel success!\n");

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    error = cuMemcpyDtoH(h_C, d_C, size_C);
    if (error != CUDA_SUCCESS)
    {
        exit_fail();
    }
    printf("copy C back success!\n");

    return 0;

}

void exit_fail()
{
    exit(1);
}



