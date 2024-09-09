#include <arm_neon.h>  // ����AVX2ָ���ͷ�ļ�
#include <stdio.h>
#include <vector>
#include <chrono>
#include <iostream>
#include "help.h"

// ������������ X ����״Ϊ (n, d_in)��Ȩ�ؾ��� W ����״Ϊ (d_in, d_out)��ƫ������ b ����״Ϊ (d_out)
void linearNEON(const float* X, const float* W, const float* b, float* output, int n, int d_in, int d_out) {
    // ����������� output ����״Ϊ (n, d_out)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d_out; j += 4) {  // ÿ�δ���8�����Ԫ��
        
            float32x4_t  sum = vmovq_n_f32(0.0f);  // ��ʼ���ۼ���Ϊ0

            for (int k = 0; k < d_in; ++k) {
                // ����Ȩ�ؾ����һ��
                float32x4_t w = vld1q_f32(&W[k * d_out + j]);
                // ��������������һ�е�һ��Ԫ��
                float32x4_t x = vmovq_n_f32(X[i * d_in + k]);
                // ����˻����ۼ�
                sum = vaddq_f32(sum, vmulq_f32(w, x));
            }

            // ����ƫ������
            float32x4_t bias = vld1q_f32(&b[j]);
            // ����ƫ��
            sum = vaddq_f32(sum, bias);

            // ������洢���������
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
    // ʾ������
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

    // ��ӡִ��ʱ��
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;

    //��ӡ������
    check(output.data(), output1.data(), n);


    return 0;
}