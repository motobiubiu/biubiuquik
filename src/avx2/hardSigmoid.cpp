#include <vector>
#include <algorithm>
#include <iostream>
#include <immintrin.h>
#include "help.h"


void hardSigmoid(const std::vector<float>& input, std::vector<float>& output) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, std::min(1.0f,(input[i]+1.0f)/2.0f));
    }
    
}

void hardSigmoidAVX(const float* input, float* output,int n) {
    
    for (int i = 0; i <= n-8; i+=8) {
        __m256 res=_mm256_set1_ps(0.0f);
        __m256 vec=_mm256_loadu_ps(&input[i]);
        __m256 one=_mm256_set1_ps(1.0f);
        __m256 two=_mm256_set1_ps(2.0f);
        __m256 div=_mm256_div_ps(_mm256_add_ps(vec, one), two);
        __m256 min_vec=_mm256_min_ps(one, div);

        res=_mm256_max_ps(res, min_vec);

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

    auto time1=measureExecutionTime(hardSigmoid,input, output1);
    auto time2=measureExecutionTime(hardSigmoidAVX,input.data(), output2.data(),n);

    // 打印时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;  
    check(output1.data(), output2.data(), n);


    return 0;
}
