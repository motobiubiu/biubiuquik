
#include <vector>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include "fun.h"
#include "help.h"

void sigmoid(const float* input,float* output,int n) {
    
    for (size_t i = 0; i < n; ++i) {
        output[i] = 1.0 / (1.0 + std::exp(-input[i]));
    }
    
}

void sigmoidAVX(const float* input,float* output,int n){    

    for(int i=0;i<n;i+=8){
        __m256 vec=_mm256_loadu_ps(&input[i]);
        __m256 res=_mm256_set1_ps(1.0f);
        __m256 one=_mm256_set1_ps(1.0f);
        __m256 neg_mask = _mm256_set1_ps(-0.0f);
        __m256 neg_vec = _mm256_xor_ps(vec, neg_mask);

        __m256 exp_vec=exp256_ps(neg_vec);
        res=_mm256_div_ps(one, _mm256_add_ps(one, exp_vec));

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

    auto time1=measureExecutionTime(sigmoid,input.data(), output1.data(),n);
    auto time2=measureExecutionTime(sigmoidAVX,input.data(), output2.data(),n);

 
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;     

    check(output1.data(), output2.data(), n);


    return 0;
}