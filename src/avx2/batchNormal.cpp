#include <iostream>
#include <vector>
#include <cmath>
#include <immintrin.h>
#include "help.h"


// 批归一化 (Batch Normalization)
void batchNormalization(float* input,float* output,float*gemma,float* beta,int batchSize,int numFeatures, float eps = 1e-5) {

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

void batchNormalizationAVX(float* input,float* output,float*gemma,float* beta,int batchSize,int numFeatures, float eps = 1e-5){
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
        for (int j = 0; j < numFeatures; j+=8) {
            __m256 vec=_mm256_loadu_ps(&input[i*numFeatures+j]);
            __m256 mean_vec=_mm256_set1_ps(mean[i]);
            __m256 stddev_vec=_mm256_set1_ps(stddev[i]);
            __m256 res=_mm256_div_ps(_mm256_sub_ps(vec, mean_vec), stddev_vec);
            _mm256_storeu_ps(&output[i*numFeatures+j], res);
        }
    }

    for (int i=0;i<batchSize;++i) {
        for (int j = 0; j < numFeatures; j+=8) {
            __m256 vec=_mm256_loadu_ps(&output[i*numFeatures+j]);
            __m256 gemma_vec=_mm256_loadu_ps(&gemma[j]);
            __m256 beta_vec=_mm256_loadu_ps(&beta[j]);
            __m256 res=_mm256_add_ps(_mm256_mul_ps(vec, gemma_vec), beta_vec);
            _mm256_storeu_ps(&output[i*numFeatures+j], res);
        }
    }
}



int main() {
    int batchSize = 1024;
    int numFeatures = 1024;

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
    auto time2=measureExecutionTime(batchNormalizationAVX,data.data(), output1.data(), gemma.data(), beta.data(), batchSize, numFeatures,1e-5);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;  
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;  


    check(output.data(), output1.data(), batchSize*numFeatures);


    return 0;
}