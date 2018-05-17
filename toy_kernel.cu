#include <stdio.h> 
// 1-D index
int __device__ getIdx_1D_1D()
{
    return blockIdx.x*blockDim.x + threadIdx.x;
}

int __device__ getIdx_1D_2D()
{
    return blockIdx.x*blockDim.x*blockDim.y 
           + threadIdx.y*blockDim.x
           + threadIdx.x;
}

int __device__ getIdx_2D_1D()
{
    int bid = blockIdx.x*gridDim.y + blockIdx.y;
    return bid*blockDim.x + threadIdx.x;
}

int __device__ getIdx_2D_2D()
{
    int bid = blockIdx.x*gridDim.y + blockIdx.y;
    return bid*blockDim.x*blockDim.y 
           + threadIdx.y*blockDim.x
           + threadIdx.x;
}
// others 1D-3D, 2D-3D, #3D-1D,2D,3D not use so much

// research row and column in cpu and gpu
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

// <<<(X,1,1),(X,1,1)>>>
// one thread one element in C
extern "C" __global__ void matrixMulv11(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    int tidx = threadIdx.x;
    int bidx = blockIdx.x;

    int idx = bidx*blockDim.x + tidx;
    
    // tidx is greater than M*N
    int row = idx/N;    // 前N个线程算第一行
    int column = idx%N; // 其中每个线程算第一行中的某一列

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

// <<<(X,Y,1),(X,Y,1)>>>
// 1-D index one thread one element in C
extern "C" __global__ void matrixMulv12(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    int idx = getIdx_2D_2D();

    // tidx is greater than M*N
    int row = idx/N;    // 前N个线程算第一行
    int column = idx%N; // 其中每个线程算第一行中的某一列
    int indexA = row*K+0;   // A 第row行的偏移量
    int stripA = 1;
    int indexB = column;    // B  第column列的偏移量
    int stripB = N;
    int indexC = row*N+column;
    
    while(row < M && column <N)
    {
        float temp = 0.f;
        for(int i = 0; i < K; ++i)
        {
            temp+=A[indexA+stripA*i]*B[indexB+stripB*i];
        }
    
        C[indexC] = temp;

       idx+=blockDim.x*blockDim.y*gridDim.x*gridDim.y;
       row = idx/N; 
       column = idx%N;
       indexA = row*K+0;
       indexB = column;
       indexC = row*N+column;
   }
}

// <<<(bX,bY,1),(tX,tY,1)>>>
// 2-D index one thread one element in C
extern "C" __global__ void matrixMulv2(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    // assume tidx*bidx > M and tidy*bidy > N

    int row_base = bidx*blockDim.x + tidx;
    int column_base = bidy*blockDim.y + tidy;
    int row = 0;
    int column = 0;
    // if matrix C small than threads, need this protect.
    // assume tidx*bidx > M and tidy*bidy > N
    for(int i = 0; i < (M/(blockDim.x*gridDim.x) + 1); ++i)
    {
        row = row_base + i*blockDim.x*gridDim.x;
        for (int j = 0; j < (N/(blockDim.y*gridDim.y) + 1); ++j)
        {
            column = column_base+j*blockDim.y*gridDim.y;
            if (row < M && column<N)
            {
                int indexA = row*K+0;   // A 第row行的偏移量
                int stripA = 1;
                int indexB = column;    // B  第column列的偏移量
                int stripB = N;
                int indexC = row*N+column;

                float temp = 0.f;
                for(int k = 0; k < K; ++k)
                {
                    temp+=A[indexA+stripA*k]*B[indexB+stripB*k];
                }
                
                C[indexC] = temp; 

                // avoid Matrix is greater than threads
                //row+=blockDim.x*gridDim.x;
                //column+=blockDim.y*gridDim.y;
            }
        }
    }
}

// <<<(1,1,1),(tX,tY,1)>>>
// 2-D index one thread one element in C
extern "C" __global__ void matrixMulv21(const float *A, const float *B, float *C, const int M, const int N, const int K)
{
    int tidx = threadIdx.x;
    int tidy = threadIdx.y;

    // assume tidx > M and tidy> N
    int row = tidx;
    int column = tidy;

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
