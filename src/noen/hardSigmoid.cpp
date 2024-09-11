#include <vector>
#include <algorithm>
#include <iostream>
#include <arm_neon.h> 
#include "help.h"
#include "funNeon.h"


void hardSigmoid(const std::vector<float>& input, std::vector<float>& output) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::max(0.0f, std::min(1.0f,(input[i]+1.0f)/2.0f));
    }
    
}

void hardSigmoidNEON(const float* input, float* output,int n) {
    
    for (int i = 0; i <= n-4; i+=4) {
        float32x4_t res=vmovq_n_f32(0.0f);
        float32x4_t vec=vld1q_f32(&input[i]);
        float32x4_t one=vmovq_n_f32(1.0f);
        float32x4_t two=vmovq_n_f32(2.0f);
        float32x4_t div=div_ps(vaddq_f32(vec, one), two);
        float32x4_t min_vec=vminq_f32(one, div);

        res=vmaxq_f32(res, min_vec);

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

    auto time1=measureExecutionTime(hardSigmoid,input, output1);
    auto time2=measureExecutionTime(hardSigmoidNEON,input.data(), output2.data(),n);


    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;  
    check(output1.data(), output2.data(), n);


    return 0;
}
