#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <immintrin.h>
#include "fun.h"
#include "help.h"


void softmax(const float* input,float* output,int n,float max_val) {
    
    float sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val); 
        sum += output[i];
    }
    for (size_t i = 0; i < n; ++i) {
        output[i] /= sum;
    }
    
}

void softmaxAVX(const float* input,float* output,int n,float max_val){    

    
    for(int i=0;i<n;i+=8){
        __m256 vec=_mm256_loadu_ps(&input[i]);
        __m256 max_vec=_mm256_set1_ps(max_val);
        vec=_mm256_sub_ps(vec, max_vec);

        __m256 res=exp256_ps(vec);
        _mm256_storeu_ps(&output[i], res);
    }
    float sum=0.0f;
    for(int i=0;i<n;++i){
        sum+=output[i];
    }
    for(int i=0;i<n;++i){
        __m256 vec=_mm256_loadu_ps(&output[i]);
        __m256 max_vec=_mm256_set1_ps(sum);

        __m256 res=_mm256_div_ps(vec, max_vec);
        _mm256_storeu_ps(&output[i], res);
    }


}


int main(){

    int n=1024*1024;
    std::vector<float> input(n,0);
    for(int i=0;i<n;++i){
        input[i]=i;
    }

    std::vector<float> output1(n,0);   
    std::vector<float> output2(n,0);
    float max_val = *std::max_element(input.begin(), input.end()); // ��ֹ���   

    auto time1=measureExecutionTime(softmax,input.data(), output1.data(),n,max_val);
    auto time2=measureExecutionTime(softmaxAVX,input.data(), output2.data(),n,max_val);

 
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;     

    check(output1.data(), output2.data(), n);


    return 0;
}