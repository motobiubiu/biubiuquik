#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <immintrin.h>
#include "help.h"
#include "fun.h"

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

void crossEntropyAVX(const float* y_true, const float* y_pred,float& output,const int n) {

    float tmp[8];
    __m256 vec_sum=_mm256_set1_ps(0.0f);
    for (int i = 0; i < n; i+=8) {
        __m256 vec_true=_mm256_loadu_ps(&y_true[i]);
        __m256 vec_pred=_mm256_loadu_ps(&y_pred[i]);
        __m256 vec_one=_mm256_set1_ps(1.0f);
        __m256 vec_x=_mm256_mul_ps(vec_true, log256_ps(vec_pred));
        __m256 vec_y=_mm256_mul_ps(_mm256_sub_ps(vec_one, vec_true), log256_ps(_mm256_sub_ps(vec_one, vec_pred)));
        __m256 vec_res=_mm256_add_ps(vec_x, vec_y);

        vec_sum=_mm256_add_ps(vec_sum, vec_res);
    }

    _mm256_storeu_ps(tmp, vec_sum);
    output= -(tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]) / (float)n;

    
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
    auto time2=measureExecutionTime(crossEntropyAVX,y_true.data(), y_pred.data(), output2,n);
    std::cout << "Cross-Entropy: " << output1 <<" time:" <<time1<< std::endl;
    std::cout << "Cross-Entropy: " << output2 <<" time:" <<time2<< std::endl;
    
    return 0;
}