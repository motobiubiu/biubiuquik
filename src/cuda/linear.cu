#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "util.h"

__global__ void matrixMulKernel(float* A, float* B, float* bias,float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum+bias[col];
    }
}

void matrixMul(float* h_A, float* h_B,float* h_bias, float* h_C, int M, int N, int K) {
    // 分配设备内存
    float *d_A, *d_B, *d_C,*d_bias;
    cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaMalloc((void**)&d_C, M * N * sizeof(float));
    cudaMalloc((void**)&d_bias, M* sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, M * sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块和网格大小
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // 启动核函数
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_bias,d_C, M, N, K);

    // 将结果从设备复制回主机
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void normal_linear(const float* X, const float* W, const float* b, float* output, int n, int d_in, int d_out){
    for(int i=0;i<n;++i){
        for(int j=0;j<d_out;++j){
            float sum=0;
            for(int k=0;k<d_in;++k){
                // sum+=X[i][k]*W[k][j];
                sum+=X[i*d_in+k]*W[k*d_out+j];
            }
            output[i*d_out+j]=sum+b[j];
        }
    }
}

int main() {
    
    std::vector<float> X;
    std::vector<float> W;
    std::vector<float> b;

    int n=2048;
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            X.push_back(j);
            W.push_back(j);
        }
        b.push_back(i);
        
    }

    std::vector<float> output(n*n,0);
    std::vector<float> output1(n*n,0);

    auto time1=measureExecutionTime(normal_linear,X.data(), W.data(), b.data(), output.data(), n, n, n);
    auto time2=measureExecutionTime(matrixMul,X.data(), W.data(), b.data(), output1.data(), n, n, n);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;

    //打印输出结果
    check(output.data(), output1.data(), n);

    return 0;
}