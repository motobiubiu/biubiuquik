#include <vector>
#include <cmath>
#include <iostream>
#include <arm_neon.h> 
#include "help.h"
#include "funNeon.h"



void elu(const std::vector<float>& input, std::vector<float>& output,float alpha) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i]>0?input[i]:alpha*(std::exp(input[i])-1);
    }
    
}

void eluNEON(const float* input, float* output,int n,float alpha) {
    
    for (int i = 0; i <= n-4; i+=4) {
        float32x4_t vec_zero=vmovq_n_f32(0.0f);
        float32x4_t vec=vld1q_f32(&input[i]);
        float32x4_t vec_exp=exp_ps(vec);

        uint32x4_t  vec_maskless=vcltq_f32(vec, vec_zero);
        float32x4_t vec_res=vbslq_f32(vec_maskless,vmulq_f32(vmovq_n_f32(alpha), vsubq_f32(vec_exp, vmovq_n_f32(1.0f))), vec);

        vst1q_f32(&output[i], vec_res);
    }
    
}


int main(){

    int n=1024;
    std::vector<float> input(n,0);
    for(int i=0;i<n;++i){
        input[i]=i;
    }
    for(int i=0;i<20;++i){
        input[i]=-i;
    }
    

    std::vector<float> output1(n,0);   
    std::vector<float> output2(n,0);   
    float alpha=0.5f;
    
    auto time1=measureExecutionTime(elu,input,output1,alpha);
    auto time2=measureExecutionTime(eluNEON,input.data(),output2.data(),n,alpha);


    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;     

    check(output1.data(), output2.data(), n);


    return 0;
}
