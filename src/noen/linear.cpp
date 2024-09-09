#include <arm_neon.h>  // 包含AVX2指令的头文件
#include <stdio.h>
#include <vector>
#include <chrono>
#include <iostream>
#include "help.h"

// 假设输入张量 X 的形状为 (n, d_in)，权重矩阵 W 的形状为 (d_in, d_out)，偏置向量 b 的形状为 (d_out)
void linearNEON(const float* X, const float* W, const float* b, float* output, int n, int d_in, int d_out) {
    // 计算输出张量 output 的形状为 (n, d_out)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d_out; j += 4) {  // 每次处理8个输出元素
        
            float32x4_t  sum = vmovq_n_f32(0.0f);  // 初始化累加器为0

            for (int k = 0; k < d_in; ++k) {
                // 加载权重矩阵的一行
                float32x4_t w = vld1q_f32(&W[k * d_out + j]);
                // 加载输入张量的一行的一个元素
                float32x4_t x = vmovq_n_f32(X[i * d_in + k]);
                // 计算乘积并累加
                sum = vaddq_f32(sum, vmulq_f32(w, x));
            }

            // 加载偏置向量
            float32x4_t bias = vld1q_f32(&b[j]);
            // 加上偏置
            sum = vaddq_f32(sum, bias);

            // 将结果存储到输出张量
            vst1q_f32(&output[i * d_out + j], sum);
        }
    }
}

void linear(const float* X, const float* W, const float* b, float* output, int n, int d_in, int d_out){
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
    // 示例数据
    std::vector<float> X;
    std::vector<float> W;
    std::vector<float> b;

    int n=256;
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            X.push_back(j);
            W.push_back(j);
        }
        b.push_back(i);
        
    }

    // float X[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
    // float W[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
    // float b[] = {1.0, 2.0, 3.0, 4.0};
    std::vector<float> output(n*n,0);
    std::vector<float> output1(n*n,0);

    auto time1=measureExecutionTime(linear,X.data(), W.data(), b.data(), output.data(), n, n, n);
    auto time2=measureExecutionTime(linearNEON,X.data(), W.data(), b.data(), output1.data(), n, n, n);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;

    //打印输出结果
    check(output.data(), output1.data(), n);


    return 0;
}