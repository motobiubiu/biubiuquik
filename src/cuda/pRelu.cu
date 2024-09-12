#include <cmath>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include "util.h"

__global__ void pReluKernel(float* input,float* output,int N,float alpha) {
    // int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx= blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        output[idx] = input[idx]>0?input[idx]:alpha*input[idx];
    }
}

void pReluCUDA(float* h_A, float* h_B, int N,float alpha) {
    // 分配设备内存
    float *d_A, *d_B;
    cudaMalloc((void**)&d_A, N *sizeof(float));
    cudaMalloc((void**)&d_B,  N * sizeof(float));

    // 将数据从主机复制到设备
    cudaMemcpy(d_A, h_A, N* sizeof(float), cudaMemcpyHostToDevice);

    // 定义线程块和网格大小
    dim3 threadsPerBlock(16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x);

    // 启动核函数
    pReluKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B,N,alpha);

    // 将结果从设备复制回主机
    cudaMemcpy(h_B, d_B,  N * sizeof(float), cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_A);
    cudaFree(d_B);
}

void pRelu(const std::vector<float>& input, std::vector<float>& output,float alpha) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i]>0?input[i]:alpha*input[i];
    }
    
}

int main() {
    int n=1024*1024;
    std::vector<float> input(n,0);
    for(int i=0;i<n;++i){
        input[i]=i;
    }

    std::vector<float> output1(n,0);   
    std::vector<float> output2(n,0);   
    float alpha=0.5f;
    
    auto time1=measureExecutionTime(pRelu,input,output1,alpha);
    auto time2=measureExecutionTime(pReluCUDA,input.data(),output2.data(),n,alpha);

    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;     

    check(output1.data(), output2.data(), n);

    return 0;
}