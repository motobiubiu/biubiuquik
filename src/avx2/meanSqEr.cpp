#include <iostream>
#include <vector>
#include <immintrin.h>
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
    for (size_t i = 0; i <n; i+=8) {
        __m256 vec_true=_mm256_loadu_ps(&y_true[i]);
        __m256 vec_pred=_mm256_loadu_ps(&y_pred[i]);
        __m256 vec_diff=_mm256_sub_ps(vec_true, vec_pred);
        __m256 vec_res=_mm256_mul_ps(vec_diff, vec_diff);
        _mm256_storeu_ps(&tmp[i], vec_res);
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

    std::cout << "Mean Squared Error: " << output1 <<" time:" <<time1<< std::endl;
    std::cout << "Mean Squared Error: " << output2 <<" time:" <<time2<< std::endl;
    return 0;
}