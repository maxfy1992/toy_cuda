#include <stdio.h> 

extern "C" __global__ void row_col_kernel(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    int tidx = threadIdx.x;
    if (tidx == 0)
    {
        printf("M=(%d) N=(%d) K=(%d)\n", M, N, K);
        for (int i = 0; i < M; ++i)
        {
            for(int j = 0; j < K; ++j)
            {
                printf("A[%d][%d]=(%f). \n", i,j,A[i*M+j]);
            }
        }   
    }
}

extern "C" __global__ void toy_kernel(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    int tidx = threadIdx.x;
    if (tidx == 0)
    {
        printf("M=(%d) N=(%d) K=(%d)\n", M, N, K);
        for (int i = 0; i < M; ++i)
        {
            for(int j = 0; j < K; ++j)
            {
                printf("A[%d][%d]=(%f). \n", i,j,A[i*M+j]);
            }
        }
    }
}
