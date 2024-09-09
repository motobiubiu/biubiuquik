#include <vector>
#include <algorithm>
#include <iostream>
#include <arm_neon.h>
#include "help.h"

// ReLU激活函数
void relu6(const std::vector<float>& input, std::vector<float>& output) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(std::max(0.0f, input[i]),6.0f);
    }
    
}

void relu6NEON(const float* input, float* output,int n) {
    
    for (int i = 0; i <= n-4; i+=4) {
        float32x4_t res=vmovq_n_f32(0.0f);
        float32x4_t vec=vld1q_f32(&input[i]);
        float32x4_t six_vec=vmovq_n_f32(6.0f);

        res=vmaxq_f32(vmaxq_f32(res, vec),six_vec);

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

    auto time1=measureExecutionTime(relu6,input, output1);
    auto time2=measureExecutionTime(relu6NEON,input.data(), output2.data(),n);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;       

    check(output1.data(), output2.data(), n);


    return 0;
}
