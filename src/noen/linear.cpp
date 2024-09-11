#include <arm_neon.h>  // ����AVX2ָ���ͷ�ļ�
#include <stdio.h>
#include <vector>
#include <chrono>
#include <iostream>
#include "help.h"


void linearNEON(const float* X, const float* W, const float* b, float* output, int n, int d_in, int d_out) {
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d_out; j += 4) {  
        
            float32x4_t  sum = vmovq_n_f32(0.0f); 

            for (int k = 0; k < d_in; ++k) {
                
                float32x4_t w = vld1q_f32(&W[k * d_out + j]);
                
                float32x4_t x = vmovq_n_f32(X[i * d_in + k]);
                
                sum = vaddq_f32(sum, vmulq_f32(w, x));
            }

            
            float32x4_t bias = vld1q_f32(&b[j]);
            
            sum = vaddq_f32(sum, bias);

            
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


    std::vector<float> output(n*n,0);
    std::vector<float> output1(n*n,0);

    auto time1=measureExecutionTime(linear,X.data(), W.data(), b.data(), output.data(), n, n, n);
    auto time2=measureExecutionTime(linearNEON,X.data(), W.data(), b.data(), output1.data(), n, n, n);


    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;


    check(output.data(), output1.data(), n);


    return 0;
}