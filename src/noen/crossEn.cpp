#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <arm_neon.h> 
#include "help.h"
#include "funNeon.h"

void crossEntropy(const float* y_true, const float* y_pred,float& output,const int n) {

    float sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (y_pred[i] < 0 || y_pred[i] > 1) {
            throw std::invalid_argument("Predicted probabilities must be in the range (0, 1).");
        }
        sum += y_true[i] * std::log(y_pred[i]) + (1 - y_true[i]) * std::log(1 - y_pred[i]);
    }
    output= -sum / n;
}

void crossEntropyNEON(const float* y_true, const float* y_pred,float& output,const int n) {

    // float sum = 0.0;
    // float tmp2[n];
    float tmp[4];
    int num=n;
    float32x4_t vec_sum=vdupq_n_f32(0.0f);
    for (int i = 0; i < n; i+=4) {
        float32x4_t vec_true=vld1q_f32(&y_true[i]);
        float32x4_t vec_pred=vld1q_f32(&y_pred[i]);
        float32x4_t vec_one=vdupq_n_f32(1.0f);
        float32x4_t vec_x=vmulq_f32(vec_true, log_ps(vec_pred));
        float32x4_t vec_y=vmulq_f32(vsubq_f32(vec_one, vec_true), log_ps(vsubq_f32(vec_one, vec_pred)));
        float32x4_t vec_res=vaddq_f32(vec_x, vec_y);

        vec_sum=vaddq_f32(vec_sum, vec_res);
        // _mm256_storeu_ps(&tmp2[i], vec_res);
        // sum += y_true[i] * std::log(y_pred[i]) + (1 - y_true[i]) * std::log(1 - y_pred[i]);
    }
    // for(int i=0;i<n;++i){
        // sum+=tmp2[i];
    // }
    vst1q_f32(tmp, vec_sum);
    output= -(tmp[0]+tmp[1]+tmp[2]+tmp[3]) / (float)num;
    // float out=-sum/n;
    
}

int main() {
    int n=2048;
    std::vector<float> y_true(n);
    std::vector<float> y_pred(n);
    float output1;
    float output2;
    for(int i=0;i<n;++i){
        y_true[i]=0.1;
        y_pred[i]=0.2;
    }

    auto time1=measureExecutionTime(crossEntropy,y_true.data(), y_pred.data(), output1,n);
    auto time2=measureExecutionTime(crossEntropyNEON,y_true.data(), y_pred.data(), output2,n);
    std::cout << "Cross-Entropy: " << output1 <<" time:" <<time1<< std::endl;
    std::cout << "Cross-Entropy: " << output2 <<" time:" <<time2<< std::endl;
    
    // crossEntropy(y_true.data(), y_pred.data(),output1,n);
    // crossEntropyAVX(y_true.data(), y_pred.data(),output2,n);
    // std::cout << "Cross-Entropy: " << output1 << std::endl;
    // std::cout << "Cross-Entropy: " << output2 << std::endl;
    return 0;
}