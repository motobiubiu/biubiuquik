#include <vector>
#include <algorithm>
#include <iostream>
#include <immintrin.h>
#include "help.h"


void relu(const std::vector<float>& input, std::vector<float>& output) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::min(std::max(0.0f, input[i]),6.0f);
    }
    
}

void reluAVX(const float* input, float* output,int n) {
    
    for (int i = 0; i <= n-8; i+=8) {
        __m256 res=_mm256_set1_ps(0.0f);
        __m256 vec=_mm256_loadu_ps(&input[i]);
        __m256 six_vec=_mm256_set1_ps(6.0f);

        res=_mm256_min_ps(_mm256_max_ps(res, vec),six_vec);

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

    auto time1=measureExecutionTime(relu,input, output1);
    auto time2=measureExecutionTime(reluAVX,input.data(), output2.data(),n);


    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;       

    check(output1.data(), output2.data(), n);


    return 0;
}
