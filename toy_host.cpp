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
void exitFail();
void matricMulv1(const float *a, const float *b, float *c, const int m , const int n, const int l);
void matricMulv2(const float *a, const float *b, float *c, const int m , const int n, const int l);
void matricMulv3(const float *a, const float *b, float *c, const int m , const int n, const int l);
void showMatrix(const char *name, const float *a, const int m , const int n);
void verify(const float *C_cpu, const float *C_gpu, int M, int N);
void randomInit(float *a, int n);

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
    int M = 8*2;//atoi(argv[1]);
    int N = 16*2;//atoi(argv[2]);
    int K = 100;//atoi(argv[3]);
    char *cubin_file = "toy_kernel.cubin";//argv[4];
    char *func_name = "matrixMulv22"; // "row_col_kernel"  
 
    CUresult error;
    
    // Initialize
    checkCudaErrors(cuInit(0));
    printf("--initialize success!\n");

    // Get number of devices supporting CUDA
    int deviceCount = 0;
    error = cuDeviceGetCount(&deviceCount);
    if (error != CUDA_SUCCESS)
    {
        exitFail();
    }
    if (deviceCount == 0)
    {
        printf("There is no device supporting CUDA.\n");
        exitFail();
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
        exitFail();
    }
    printf("--getDevice success\n");

   // Create context
    error = cuCtxCreate(&cuContext, 0, cuDevice);
    if (error != CUDA_SUCCESS)
    {
        exitFail();
    }
    printf("--create context success!\n");

    //load cubin module
    error = cuModuleLoad(&cuModule, cubin_file);
    if (error != CUDA_SUCCESS)
    {
        exitFail();
    }
    printf("--load cubin success!\n");

    error = cuModuleGetFunction(&toy_kernel, cuModule, func_name);
    if (error != CUDA_SUCCESS)
    {
        exitFail();
    }
    printf("--get function success!\n");    

    size_t size_A = M * K * sizeof(float);
    // Allocate input  h_A and h_B in host memory
    h_A = (float *)malloc(size_A);
    if (h_A == NULL)
    {
        exitFail();;
    }
    printf("--alloc h_A success!\n");

    size_t size_B = K * N * sizeof(float);
    h_B = (float *)malloc(size_B);
    if (h_B == NULL)
    {
        exitFail();
    }
    printf("--alloc h_B success!\n");

    size_t size_C = N * M * sizeof(float);
    h_C = (float *)malloc(size_C);
    if (h_C == NULL)
    {
        exitFail();
    }
    printf("--alloc h_C success!\n");

    h_C_cpu = (float *)malloc(size_C);
    if (h_C_cpu == NULL)
    {
        exitFail();
    }
    printf("--alloc h_C_cpu success!\n");

    // Allocate in device memory
    error = cuMemAlloc(&d_A, size_A);
    if (error != CUDA_SUCCESS)
    {
        exitFail();
    }
    printf("--alloc d_A success!\n");
    error = cuMemAlloc(&d_B, size_B); 
    if (error != CUDA_SUCCESS)
    {
        exitFail();
    }
    printf("--alloc d_B success!\n");

    error = cuMemAlloc(&d_C, size_C); 
    if (error != CUDA_SUCCESS)
    {
        exitFail();
    }
    printf("--alloc d_C success!\n");
    
    // init input
    for(int i=0;i<M*N;i++)
    {
       h_C[i] = 0;
       h_C_cpu[i] = 0;
    }

    randomInit(h_A, M*K);
    randomInit(h_B, N*K);
    /* //print init values
    showMatrix("A", h_A, M, K);
    showMatrix("B", h_B, K, N);
    showMatrix("C", h_C, M, N);
    showMatrix("C_cpu", h_C_cpu, M, N);
    */ 
    /* // compute matric C
    matricMulv1(h_A, h_B, h_C_cpu, M , N, K);
    showMatrix("C_cpu", h_C_cpu, M, N);
    for(int i=0;i<M*N;i++)
    {
       h_C_cpu[i] = 0;
    }
    matricMulv2(h_A, h_B, h_C_cpu, M , N, K);
    showMatrix("C_cpu", h_C_cpu, M, N);
    for(int i=0;i<M*N;i++)
    {
       h_C_cpu[i] = 0;
    }
    */
    matricMulv3(h_A, h_B, h_C_cpu, M , N, K);
   // showMatrix("C_cpu", h_C_cpu, M, N);
    
    error = cuMemcpyHtoD(d_A, h_A, size_A);
    if (error != CUDA_SUCCESS)
    {
        exitFail(); 
    }
    printf("--memcpy h_A to devivce d_A  success!\n");

    error = cuMemcpyHtoD(d_B, h_B, size_B);
    if (error != CUDA_SUCCESS)
    {
        exitFail();
    }
    printf("--memcpy h_B to devivce d_B  success!\n");

    error = cuMemcpyHtoD(d_C, h_C, size_C);
    if (error != CUDA_SUCCESS)
    {
        exitFail();
    }
    printf("--memcpy h_C to devivce d_C  success!\n");

    void *args[] = {&d_A, &d_B, &d_C, &M, &N, &K};
    
    //Launch the CUDA kernel
    checkCudaErrors(cuLaunchKernel(toy_kernel, 8, 1, 1,
                           16, 1, 1, 
                           0,
                           NULL, args, NULL));
    printf("--launch kernel success!\n");

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    error = cuMemcpyDtoH(h_C, d_C, size_C);
    if (error != CUDA_SUCCESS)
    {
        exitFail();
    }
    printf("--copy C back success!\n");
    //showMatrix("C", h_C, M, N);
    verify(h_C_cpu, h_C, M,N);
    return 0;

}

void exitFail()
{
    exit(1);
}

// a m*l
// b l*n
// c m*n
// b 访存不连续，无法向量化
void matricMulv1(const float *a, const float *b, float *c, const int m , const int n, const int l)
{
    for (int i = 0; i < m; ++i)
    {
      for (int j = 0; j < n; ++j)
      {
          for (int k = 0; k < l; ++k)
          {
              // 2-dims
              //c[i][j]+= a[i][k]*b[k][j];
              // 1-dims
              c[i*n+j] += a[i*l+k]*b[k*n+j];
          }
      }
    }
}

// 交换内存循环，使得b访存连续，可以向量化
// a m*l
// b l*n
// c m*n
void matricMulv2(const float *a, const float *b, float *c, const int m , const int n, const int l)
{
    for (int i = 0; i < m; ++i)
    {
      for (int k = 0; k < l; ++k)
      {
          //temp = a[i][k];
          float temp = a[i*l+k];
          for (int j = 0; j < n; ++j)
          {
              // 2-dims
              //c[i][j]+= temp*b[k][j];
              // 1-dims
              c[i*n+j] += temp*b[k*n+j];
          }
      }
    }
}

// 转置b矩阵，使得b访存连续，可以向量化
// a m*l
// b l*n
// c m*n
void matricMulv3(const float *a, const float *b, float *c, const int m , const int n, const int l)
{
  float *b1=(float*)malloc(sizeof(float)*l*n);
  memset(b1, 0, sizeof(float)*l*n);
  for (int i = 0; i < l; ++i)
  {
      for (int j = 0; j < n; ++j)
      {
          //b1[i][j] = b[j][i];
          b1[j*l+i] = b[i*n+j];
      }
  }
//showMatrix("B'", b1, n, l);
  for (int i = 0; i < m; ++i)
  {
      for (int j = 0; j < n; ++j)
      {
          for (int k = 0; k < l; ++k)
          {
              // 2-dims
              //c[i][j]+= a[i][k]*b1[j][k];
              // 1-dims
              c[i*n+j] += a[i*l+k]*b1[j*l+k];
          }
      }
  }

  free(b1);
}

void showMatrix(const char *name, const float *a, const int m , const int n)
{
    printf("show %s \n", name);
    for (int i = 0; i < m; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            printf("matrix[%d][%d]=(%f). \n", i,j, a[i*n+j]);
        }
    } 
}

void verify(const float *C_cpu, const float *C_gpu, int M, int N)
{
    for(int m=0;m<M;m++)
    {
        for(int n=0;n<N;n++)
        {
            if(fabs(C_cpu[n+m*N] - C_gpu[n+m*N]) > 1e-5)
            {
		printf("Verify Failed!\n");
		printf("cpu result: %f, gpu result: %f\n",C_cpu[n+m*N],C_gpu[n+m*N]);
		exit(1);
            }
        }
    }
    printf("Verify Success!\n");
}

void randomInit(float *a, int n)
{
    for (int i = 0; i < n; ++i)
    {
        a[i] = rand() / (float)RAND_MAX;
    }
}
