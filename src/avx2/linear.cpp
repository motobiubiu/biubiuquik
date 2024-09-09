#include <immintrin.h>  // ����AVX2ָ���ͷ�ļ�
#include <vector>
#include <iostream>
#include "help.h"

// 假设输入张量 X 的形状为 (n, d_in)，权重矩阵 W 的形状为 (d_in, d_out)，偏置向量 b 的形状为 (d_out)
void normal_linearAVX(const float* X, const float* W, const float* b, float* output, int n, int d_in, int d_out) {
    // 计算输出张量 output 的形状为 (n, d_out)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d_out; j += 8) {  // 每次处理8个输出元素
            __m256 sum = _mm256_setzero_ps();  // 初始化累加器为0

            for (int k = 0; k < d_in; ++k) {
                // 加载权重矩阵的一行
                __m256 w = _mm256_loadu_ps(&W[k * d_out + j]);
                // 加载输入张量的一行的一个元素
                __m256 x = _mm256_set1_ps(X[i * d_in + k]);
                // 计算乘积并累加
                sum = _mm256_add_ps(sum, _mm256_mul_ps(w, x));
            }

            // 加载偏置向量
            __m256 bias = _mm256_loadu_ps(&b[j]);
            // 加上偏置
            sum = _mm256_add_ps(sum, bias);

            // 将结果存储到输出张量
            _mm256_storeu_ps(&output[i * d_out + j], sum);
        }
    }
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
    auto time2=measureExecutionTime(normal_linearAVX,X.data(), W.data(), b.data(), output1.data(), n, n, n);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;

    //打印输出结果
    check(output.data(), output1.data(), n);
    return 0;
}