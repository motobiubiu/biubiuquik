#include <vector>
#include <iostream>
#include <arm_neon.h> 
#include "help.h"
#include "funNeon.h"

// hardWish激活函数
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

void hardWishNEON(const float* input, float* output,int n) {
    
    for (int i = 0; i <= n-4; i+=4) {
        float32x4_t res=vdupq_n_f32(0.0f);
        float32x4_t vec=vld1q_f32(&input[i]);
        
        uint32x4_t  maskless=vcltq_f32(vec, vdupq_n_f32(-3.0f));
        uint32x4_t  maskgreat=vcgtq_f32(vec, vdupq_n_f32(3.0f));
        
        res=vbslq_f32(
                maskless,
                vdupq_n_f32(0.0f),
                vbslq_f32( maskgreat,
                           vec,
                           vmulq_f32(vec, div_ps(vaddq_f32(vec, vdupq_n_f32(3.0f)), vdupq_n_f32(6.0f)))                           
                           )                
                 );

        // res=_mm256_max_ps(res, vec);

        vst1q_f32(&output[i], res);
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
    auto time2=measureExecutionTime(hardWishNEON,input.data(), output2.data(),n);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;

    check(output1.data(), output2.data(), n);     

    return 0;
}
