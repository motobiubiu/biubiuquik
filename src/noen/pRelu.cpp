#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <arm_neon.h>  
#include <chrono>
#include "help.h"

// pRelu激活函数
void pRelu(const std::vector<float>& input, std::vector<float>& output,float alpha) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i]>0?input[i]:alpha*input[i];
    }
    
}

void pReluNEON(const float* input, float* output,int n,float alpha) {
    
    for (int i = 0; i <= n-4; i+=4) {
        float32x4_t vec_zero=vdupq_n_f32(0.0f);
        float32x4_t vec=vld1q_f32(&input[i]);

        uint32x4_t vec_maskless=vcltq_f32(vec, vec_zero);
        float32x4_t vec_res=vbslq_f32(vec_maskless, vmulq_f32(vdupq_n_f32(alpha), vec),vec);



        vst1q_f32(&output[i], vec_res);
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
    auto time2=measureExecutionTime(pReluNEON,input.data(),output2.data(),n,alpha);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;     

    check(output1.data(), output2.data(), n);


    return 0;
}
