#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <immintrin.h>
#include "help.h"
#include "fun.h"


void klDivergence(const float* y_true, const float* y_pred,float& output,const int n) {

    // if (p.size() != q.size()) {
    //     throw std::invalid_argument("Input vectors must have the same size.");
    // }
    float sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (y_true[i] <= 0 || y_pred[i] <= 0) {
            throw std::invalid_argument("Probabilities must be positive.");
        }
        sum += y_true[i] * std::log(y_true[i] / y_pred[i]);
    }
    output=sum;
}

void klDivergenceAVX(const float* y_true, const float* y_pred,float& output,const int n) {

    float sum = 0.0;
    float tmp[8];
    __m256 vec_sum=_mm256_set1_ps(0.0f);
    for (size_t i = 0; i <n; i+=8) {
        __m256 vec_true=_mm256_loadu_ps(&y_true[i]);
        __m256 vec_pred=_mm256_loadu_ps(&y_pred[i]);
        __m256 vec_log=log256_ps(_mm256_div_ps(vec_true, vec_pred));
        __m256 vec_res=_mm256_mul_ps(vec_true, vec_log);
        vec_sum=_mm256_add_ps(vec_sum, vec_res);
        // _mm256_storeu_ps(&tmp[i], vec_res);
    }
    _mm256_storeu_ps(tmp, vec_sum);
    // for(int i=0;i<n;++i){
    //     sum+=tmp[i];
    // }
    sum=tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
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
    auto time2=measureExecutionTime(klDivergenceAVX,y_true.data(), y_pred.data(), output2,n);

    std::cout << "Mean Squared Error: " << output1 <<" time:" <<time1<< std::endl;
    std::cout << "Mean Squared Error: " << output2 <<" time:" <<time2<< std::endl;
    return 0;
}