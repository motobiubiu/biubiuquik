#include <iostream>
#include <vector>
#include <arm_neon.h>  
#include "help.h"


void meanSquaredError(const float* y_true, const float* y_pred,float& output,const int n) {

    float sum = 0.0;
    for (size_t i = 0; i <n; ++i) {
        float diff = y_true[i] - y_pred[i];
        sum += diff * diff;
    }
    output= sum / (float)n;
}

void meanSquaredErrorAVX(const float* y_true, const float* y_pred,float& output,const int n) {

    float sum = 0.0;
    float tmp[n];
    for (size_t i = 0; i <n; i+=4) {
        float32x4_t vec_true=vld1q_f32(&y_true[i]);
        float32x4_t vec_pred=vld1q_f32(&y_pred[i]);
        float32x4_t vec_diff=vsubq_f32(vec_true, vec_pred);
        float32x4_t vec_res=vmulq_f32(vec_diff, vec_diff);
        vst1q_f32(&tmp[i], vec_res);
    }
    for(int i=0;i<n;++i){
        sum+=tmp[i];
    }
    output= sum / (float)n;
}

int main() {
    int n=1024*1024;
    std::vector<float> y_true (n);
    std::vector<float> y_pred (n);
    float output1=0.f;
    float output2=0.f;
    for(int i=0;i<n;++i){
        y_true[i]=i;
        y_pred[i]=i+0.5f;
    }
    auto time1=measureExecutionTime(meanSquaredError,y_true.data(), y_pred.data(), output1,n);
    auto time2=measureExecutionTime(meanSquaredErrorAVX,y_true.data(), y_pred.data(), output2,n);
    // meanSquaredError(y_true.data(), y_pred.data(), output1,n);
    // meanSquaredErrorAVX(y_true.data(), y_pred.data(), output2,n);
    std::cout << "Mean Squared Error: " << output1 <<" time:" <<time1<< std::endl;
    std::cout << "Mean Squared Error: " << output2 <<" time:" <<time2<< std::endl;
    return 0;
}