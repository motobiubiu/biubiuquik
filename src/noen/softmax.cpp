#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <arm_neon.h>
#include "funNeon.h"
#include "help.h"


void softmax(const float* input,float* output,int n,float max_val) {
    
    float sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        output[i] = std::exp(input[i] - max_val); // 减去最大值防止溢出
        sum += output[i];
    }
    for (size_t i = 0; i < n; ++i) {
        output[i] /= sum;
    }
    
}

void softmaxNEON(const float* input,float* output,int n,float max_val){    

    
    for(int i=0;i<n;i+=4){
        float32x4_t vec=vld1q_f32(&input[i]);
        float32x4_t max_vec=vmovq_n_f32(max_val);
        vec=vsubq_f32(vec, max_vec);

        float32x4_t res=exp_ps(vec);
        vst1q_f32(&output[i], res);
    }
    float sum=0.0f;
    for(int i=0;i<n;++i){
        sum+=output[i];
    }
    for(int i=0;i<n;i+=4){
        float32x4_t vec=vld1q_f32(&output[i]);
        float32x4_t max_vec=vmovq_n_f32(sum);

        float32x4_t res=div_ps(vec, max_vec);
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
    float max_val = *std::max_element(input.begin(), input.end()); // 防止溢出   

    auto time1=measureExecutionTime(softmax,input.data(), output1.data(),n,max_val);
    auto time2=measureExecutionTime(softmaxNEON,input.data(), output2.data(),n,max_val);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;     

    check(output1.data(), output2.data(), n);


    return 0;
}