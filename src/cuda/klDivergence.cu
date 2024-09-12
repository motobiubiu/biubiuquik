#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "util.h"


__global__ void klDivergenceKernel(float* A,float* B,float* output,int N) {
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    int tid=threadIdx.x;
    extern __shared__ float sharedData[];

    if (idx < N) {
        sharedData[tid]=A[idx]*logf(A[idx]/B[idx]);
    }

    // 使用归约算法进行累加
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sharedData[0]);
    }
}

void klDivergenceCUDA(float* h_A, float* h_B, float& output, int N) {
    // 分配设备内存
    float *d_A, *d_B, *d_output;
    cudaMalloc((void**)&d_A, N *sizeof(float));
    cudaMalloc((void**)&d_B,  N * sizeof(float));
    cudaMalloc((void**)&d_output,  sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, N* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N* sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块和网格大小
    dim3 threadsPerBlock(16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // 启动核函数
    klDivergenceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B,d_output,N);

    // 将结果从设备复制回主机
    cudaMemcpy(&output, d_output,  sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
}

void klDivergence(const float* y_true, const float* y_pred,float& output,const int n) {

    float sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (y_true[i] <= 0 || y_pred[i] <= 0) {
            throw std::invalid_argument("Probabilities must be positive.");
        }
        sum += y_true[i] * std::log(y_true[i] / y_pred[i]);
    }
    output=sum;
}


int main() {
    int n=1024*1024;
    std::vector<float> y_true (n);
    std::vector<float> y_pred (n);
    float output1=0.f;
    float output2=0.f;
    for(int i=0;i<n;++i){
        y_true[i]=i+0.1f;
        y_pred[i]=i+0.5f;
    }
    auto time1=measureExecutionTime(klDivergence,y_true.data(), y_pred.data(), output1,n);
    auto time2=measureExecutionTime(klDivergenceCUDA,y_true.data(), y_pred.data(), output2,n);

    std::cout << "Mean Squared Error: " << output1 <<" time:" <<time1<< std::endl;
    std::cout << "Mean Squared Error: " << output2 <<" time:" <<time2<< std::endl;
    return 0;
}