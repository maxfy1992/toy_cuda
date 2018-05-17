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

// <<<(1,1,1),(32,1,1)>>>
extern "C" __global__ void matrixMulv1(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    int tidx = threadIdx.x; // 0 - 31
    // tidx is greater than M*N
    int row = tidx/N;    // 前N个线程算第一行
    int column = tidx%N; // 其中每个线程算第一行中的某一列

    int indexA = row*K+0;   // A 第row行的偏移量
    int stripA = 1;
    int indexB = column;    // B  第column列的偏移量
    int stripB = N;
    int indexC = row*N+column;

    float temp = 0.f;
    for(int i = 0; i < K; ++i)
    {
        temp+=A[indexA+stripA*i]*B[indexB+stripB*i];
    }
    
    C[indexC] = temp;
    
}
