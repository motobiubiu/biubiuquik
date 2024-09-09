#include <iostream>
#include <vector>
#include <cmath>
#include <arm_neon.h> 
#include "help.h"
#include "funNeon.h"


// 批归一化 (Batch Normalization)
void batchNormalization(float* input,float* output,float*gemma,float* beta,int batchSize,int numFeatures, float eps = 1e-5) {
    // int batchSize = x.size();
    // int numFeatures = x[0].size();

    std::vector<float> mean(batchSize, 0.0);
    std::vector<float> variance(batchSize, 0.0);
    std::vector<float> stddev(batchSize, 0.0);

    // 计算均值
    for (int i=0;i<batchSize;++i) {
        for (int j = 0; j < numFeatures; ++j) {
            mean[i] += input[i*numFeatures+j];
        }
        mean[i]/=numFeatures;
    }


    // 计算方差和标准差
    for (int i=0;i<batchSize;++i) {
        for (int j = 0; j < numFeatures; ++j) {
            variance[i]+=(input[i*numFeatures+j]-mean[i])*(input[i*numFeatures+j]-mean[i]);
        }
        variance[i]/=numFeatures;
    }
    for (int j = 0; j < batchSize; ++j) {
        stddev[j] = std::sqrt(variance[j] + eps);
    }

    // 归一化
    for (int i=0;i<batchSize;++i) {
        for (int j = 0; j < numFeatures; j++) {
            output[i*numFeatures+j]=(input[i*numFeatures+j]-mean[i])/stddev[i];
        }
    }

    for (int i=0;i<batchSize;++i) {
        for (int j = 0; j < numFeatures; j++) {
            output[i*numFeatures+j]=output[i*numFeatures+j]*gemma[j]+beta[j];
        }
    }
}

void batchNormalizationNEON(float* input,float* output,float*gemma,float* beta,int batchSize,int numFeatures, float eps = 1e-5){
    std::vector<float> mean(batchSize, 0.0);
    std::vector<float> variance(batchSize, 0.0);
    std::vector<float> stddev(batchSize, 0.0);


    // 计算均值
    for (int i=0;i<batchSize;++i) {
        for (int j = 0; j < numFeatures; ++j) {
            mean[i] += input[i*numFeatures+j];
        }
        mean[i]/=numFeatures;
    }


    // 计算方差和标准差
    for (int i=0;i<batchSize;++i) {
        for (int j = 0; j < numFeatures; ++j) {
            variance[i]+=(input[i*numFeatures+j]-mean[i])*(input[i*numFeatures+j]-mean[i]);
        }
        variance[i]/=numFeatures;
    }
    for (int j = 0; j < batchSize; ++j) {
        stddev[j] = std::sqrt(variance[j] + eps);
    }

    // 归一化
    for (int i=0;i<batchSize;++i) {
        for (int j = 0; j < numFeatures; j+=4) {
            float32x4_t vec=vld1q_f32(&input[i*numFeatures+j]);
            float32x4_t mean_vec=vmovq_n_f32(mean[i]);
            float32x4_t stddev_vec=vmovq_n_f32(stddev[i]);
            float32x4_t res=div_ps(vsubq_f32(vec, mean_vec), stddev_vec);
            vst1q_f32(&output[i*numFeatures+j], res);
            // output[i*numFeatures+j]=(input[i*numFeatures+j]-mean[i])/stddev[i];
        }
    }

    for (int i=0;i<batchSize;++i) {
        for (int j = 0; j < numFeatures; j+=4) {
            float32x4_t vec=vld1q_f32(&output[i*numFeatures+j]);
            float32x4_t gemma_vec=vld1q_f32(&gemma[j]);
            float32x4_t beta_vec=vld1q_f32(&beta[j]);
            float32x4_t res=vaddq_f32(vmulq_f32(vec, gemma_vec), beta_vec);
            vst1q_f32(&output[i*numFeatures+j], res);
            // output[i*numFeatures+j]=output[i*numFeatures+j]*gemma[j]+beta[j];
        }
    }
}



int main() {
    int batchSize = 1024;
    int numFeatures = 1024;
    // std::vector<float> data = {
    //     1, 2, 3, 4, 5, 6, 7, 8,
    //     9, 10, 11, 12, 13, 14, 15, 16,
    //     17, 18, 19, 20, 21, 22, 23, 24,
    //     25, 26, 27, 28, 29, 30, 31, 32
    // };

    // std::cout << "Original Data:\n";
    // for (int i = 0; i < batchSize; ++i) {
    //     for (int j = 0; j < numFeatures; ++j) {
    //         std::cout << data[i * numFeatures + j] << " ";
    //     }
    //     std::cout << "\n";
    // }
    std::vector<float>data(batchSize*numFeatures,0);
    for(int i=0;i<batchSize;++i){
        for(int j=0;j<numFeatures;++j){
            data[i*numFeatures+j]=j;
        }
    }
    std::vector<float> output(batchSize*numFeatures,0);
    std::vector<float> output1(batchSize*numFeatures,0);
    std::vector<float> gemma(numFeatures,0);
    std::vector<float> beta(numFeatures,0);

    auto time1=measureExecutionTime(batchNormalization,data.data(), output.data(), gemma.data(), beta.data(), batchSize, numFeatures,1e-5);
    auto time2=measureExecutionTime(batchNormalizationNEON,data.data(), output1.data(), gemma.data(), beta.data(), batchSize, numFeatures,1e-5);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;  
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;  


    check(output.data(), output1.data(), batchSize*numFeatures);


    return 0;
}