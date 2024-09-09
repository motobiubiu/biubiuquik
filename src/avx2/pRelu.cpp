#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <immintrin.h>
#include <chrono>
#include "help.h"


void pRelu(const std::vector<float>& input, std::vector<float>& output,float alpha) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i]>0?input[i]:alpha*input[i];
    }
    
}

void pReluAVX(const float* input, float* output,int n,float alpha) {
    
    for (int i = 0; i <= n-8; i+=8) {
        __m256 vec_zero=_mm256_set1_ps(0.0f);
        __m256 vec=_mm256_loadu_ps(&input[i]);

        __m256 vec_maskless=_mm256_cmp_ps(vec, vec_zero, _CMP_LT_OQ);
        __m256 vec_res=_mm256_blendv_ps(vec,_mm256_mul_ps(_mm256_set1_ps(alpha), vec), vec_maskless);



        _mm256_storeu_ps(&output[i], vec_res);
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
    float alpha=0.5f;
    
    auto time1=measureExecutionTime(pRelu,input,output1,alpha);
    auto time2=measureExecutionTime(pReluAVX,input.data(),output2.data(),n,alpha);

 
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;     

    check(output1.data(), output2.data(), n);


    return 0;
}
