#include "help.h"
#include <arm_neon.h>   // ����AVXָ���ͷ�ļ�
#include <stdio.h>
#include <float.h> // ����FLT_MAX
#include <vector>
#include <iostream>

void maxPooling(const float* X, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int OH = (H - kH) / sH + 1; // 输出高度
    int OW = (W - kW) / sW + 1; // 输出宽度

    for (int h = 0; h < OH; ++h) {
        for (int w = 0; w < OW; ++w) {
            float max_value = -FLT_MAX; // 初始化最大值为负无穷

            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) {
                    int input_h = h * sH + kh;
                    int input_w = w * sW + kw;
                    float value = X[input_h * W + input_w];
                    if (value > max_value) {
                        max_value = value;
                    }
                }
            }

            output[h * OW + w] = max_value;
        }
    }
}

void maxPoolingNEON(const float* X, float* output, int H, int W, int kH, int kW, int sH, int sW) {
    int OH = (H - kH) / sH + 1; // 输出高度
    int OW = (W - kW) / sW + 1; // 输出宽度

    for (int h = 0; h < OH; ++h) {
        for (int w = 0; w < OW; w+=4) {// 每次处理8个元素
            float32x4_t max_vec = vdupq_n_f32(-FLT_MAX); // 初始化最大值向量为负无穷

            for (int kh = 0; kh < kH; ++kh) {
                for (int kw = 0; kw < kW; ++kw) { 
                    int input_h,input_w;
                    float num0=0.f,num1=0.f,num2=0.f,num3=0.f;

                    input_h = h * sH + kh;
                    input_w = w * sW + kw;
                    num0=X[input_h * W + input_w];

                    if(w+1<OW){
                    input_w = (w+1) * sW + kw;
                    num1=X[input_h * W + input_w];                        
                    }

                    if(w+2<OW){
                    input_w = (w+2) * sW + kw;
                    num2=X[input_h * W + input_w];                        
                    }

                    if(w+3<OW){
                    input_w = (w+3) * sW + kw;
                    num3=X[input_h * W + input_w];                        
                    }

                    // if(w+4<OW){
                    // input_w = (w+4) * sW + kw;
                    // num4=X[input_h * W + input_w];                        
                    // }

                    // if(w+5<OW){
                    // input_w = (w+5) * sW + kw;
                    // num5=X[input_h * W + input_w];                        
                    // }

                    // if(w+6<OW){
                    // input_w = (w+6) * sW + kw;
                    // num6=X[input_h * W + input_w];                        
                    // }

                    // if(w+7<OW){
                    // input_w = (w+7) * sW + kw;
                    // num7=X[input_h * W + input_w];                        
                    // }

                    // __m256 vec=_mm256_setr_ps(num0,num1, num2, num3, num4, num5, num6, num7);
                    float tmp[]={num0,num1,num2,num3};
                    float32x4_t vec=vld1q_f32(tmp);
                    max_vec = vmaxq_f32(max_vec, vec);
                }
            }

             vst1q_f32(&output[h * OW + w],max_vec);
        }
    }
}

int main() {
    // 示例输入特征图 (H=4, W=4)
    // float input[4 * 4] = {
    //     1, 3, 2, 4,
    //     5, 6, 1, 2,
    //     7, 8, 3, 0,
    //     2, 1, 4, 5
    // };
    int n=128;
    std::vector<float> input(n*n,0);
    for(int i=0;i<n;++i){
        for(int j=0;j<n;++j){
            input[i*n+j]=j;
        }
    }

    std::vector<float> output1(17*17,0);
    std::vector<float> output2(17*17,0);

    auto time1=measureExecutionTime(maxPooling,input.data(), output1.data(), 128, 128,64, 64, 4, 4);
    auto time2=measureExecutionTime(maxPoolingNEON,input.data(), output2.data(), 128, 128,64, 64, 4, 4);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;
    check(output1.data(), output2.data(), 17*17);


    return 0;
}