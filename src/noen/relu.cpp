#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <arm_neon.h>
#include <chrono>
#include "help.h"

// ReLU激活函数
void relu(const std::vector<float>& input, std::vector<float>& output) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, input[i]);
    }
    
}

void reluNEON(const float* input, float* output,int n) {
    
    for (int i = 0; i <= n-4; i+=4) {
        float32x4_t res=vdupq_n_f32(0.0f);
        float32x4_t vec=vld1q_f32(&input[i]);

        res=vmaxq_f32(res, vec);

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

    auto time1=measureExecutionTime(relu,input, output1);
    auto time2=measureExecutionTime(reluNEON,input.data(), output2.data(),n);

    // 打印执行时间
    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;       

    check(output1.data(), output2.data(), n);
   


    return 0;
}
