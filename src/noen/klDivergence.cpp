#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <arm_neon.h> 
#include "help.h"
#include "funNeon.h"


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

void klDivergenceNEON(const float* y_true, const float* y_pred,float& output,const int n) {

    float sum = 0.0;
    float tmp[4];
    float32x4_t vec_sum=vdupq_n_f32(0.0f);
    for (size_t i = 0; i <n; i+=4) {
        float32x4_t vec_true=vld1q_f32(&y_true[i]);
        float32x4_t vec_pred=vld1q_f32(&y_pred[i]);
        float32x4_t vec_log=log_ps(div_ps(vec_true, vec_pred));
        float32x4_t vec_res=vmulq_f32(vec_true, vec_log);
        vec_sum=vaddq_f32(vec_sum, vec_res);
    }
    vst1q_f32(tmp, vec_sum);

    sum=tmp[0]+tmp[1]+tmp[2]+tmp[3];
    output= sum;
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
    auto time2=measureExecutionTime(klDivergenceNEON,y_true.data(), y_pred.data(), output2,n);

    std::cout << "Mean Squared Error: " << output1 <<" time:" <<time1<< std::endl;
    std::cout << "Mean Squared Error: " << output2 <<" time:" <<time2<< std::endl;
    return 0;
}