#include <vector>
#include <cmath>
#include <iostream>
#include <arm_neon.h>
#include "funNeon.h"
#include "help.h"


void silu(const float* input,float* output,int n) {
    
    for (size_t i = 0; i < n; ++i) {
        output[i] = input[i]*(1.0 / (1.0 + std::exp(-input[i])));
    }
    
}

void siluNEON(const float* input,float* output,int n){    

    for(int i=0;i<n;i+=4){
        float32x4_t vec=vld1q_f32(&input[i]);
        float32x4_t res=vmovq_n_f32(1.0f);
        float32x4_t one=vmovq_n_f32(1.0f);

        float32x4_t exp_vec=div_ps(one,exp_ps(vec));
        res=div_ps(one, vaddq_f32(one, exp_vec));
        res=vmulq_f32(res, vec);

        vst1q_f32(&output[i], res);
    }


}


int main(){

    int n=2048*2048;
    std::vector<float> input(n,0);
    for(int i=0;i<n;++i){
        input[i]=i;
    }

    std::vector<float> output1(n,0);   
    std::vector<float> output2(n,0);   

    auto time1=measureExecutionTime(silu,input.data(), output1.data(),n);
    auto time2=measureExecutionTime(siluNEON,input.data(), output2.data(),n);

    // ��ӡִ��ʱ��
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;     

    check(output1.data(), output2.data(), n);


    return 0;
}