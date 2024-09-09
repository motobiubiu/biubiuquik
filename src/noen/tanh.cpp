#include <vector>
#include <cmath>
#include <iostream>
#include <arm_neon.h>
#include "funNeon.h"
#include "help.h"


void tanh1(const float* input,float* output,int n) {
    
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::tanh(input[i]);
    }
    
}

void tanhNEON(const float* input,float* output,int n){    

    for(int i=0;i<n;i+=4){
        float32x4_t vec=vld1q_f32(&input[i]);
        float32x4_t one=vdupq_n_f32(1.0f);
        // float32x4_t neg_mask = _mm256_set1_ps(-0.0f);
        // float32x4_t neg_vec = _mm256_xor_ps(vec, neg_mask);

        float32x4_t m1=exp_ps(vec);
        float32x4_t m2=div_ps(one,m1);

        float32x4_t res=div_ps(vsubq_f32(m1, m2), vaddq_f32(m1, m2));


        vst1q_f32(&output[i], res);
    }


}


int main(){

    int n=1024*1024;
    std::vector<float> input(n,0);
    for(int i=0;i<n;++i){
        input[i]=0.1*i;
    }

    std::vector<float> output1(n,0);   
    std::vector<float> output2(n,0);   

    auto time1=measureExecutionTime(tanh1,input.data(), output1.data(),n);
    auto time2=measureExecutionTime(tanhNEON,input.data(), output2.data(),n);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;     

    check(output1.data(), output2.data(), n);


    return 0;
}