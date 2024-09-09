#include <vector>
#include <iostream>
#include <immintrin.h>
#include "help.h"


void hardWish(const std::vector<float>& input, std::vector<float>& output) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        if(input[i]<=-3.0f){
            output[i]=0;
        }else if(input[i]>=3){
            output[i]=input[i];
        }else{
            output[i]=input[i]*(input[i]+3.0f)/6.0f;
        }
    }
    
}

void hardWishAVX(const float* input, float* output,int n) {
    
    for (int i = 0; i <= n-8; i+=8) {
        __m256 res=_mm256_set1_ps(0.0f);
        __m256 vec=_mm256_loadu_ps(&input[i]);
        
        __m256 maskless=_mm256_cmp_ps(vec, _mm256_set1_ps(-3.0f), _CMP_LT_OQ);
        __m256 maskgreat=_mm256_cmp_ps(vec, _mm256_set1_ps(3.0f), _CMP_GT_OQ);
        
        res=_mm256_blendv_ps(
                _mm256_blendv_ps(_mm256_mul_ps(vec, _mm256_div_ps(_mm256_add_ps(vec, _mm256_set1_ps(3.0f)), _mm256_set1_ps(6.0f))), 
                                    vec, 
                                    maskgreat), 
                  _mm256_set1_ps(0.0f), 
                 maskless);

        // res=_mm256_max_ps(res, vec);

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

    auto time1=measureExecutionTime(hardWish,input, output1);
    auto time2=measureExecutionTime(hardWishAVX,input.data(), output2.data(),n);

    // 打印时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;

    check(output1.data(), output2.data(), n);     

    return 0;
}
