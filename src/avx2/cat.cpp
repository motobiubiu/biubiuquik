#include <vector>
// #include <cmath>
// #include <algorithm>
#include <iostream>
#include <immintrin.h>
#include <chrono>
#include "help.h"


void cat(float* input1,float* input2, float* output,int row,int col,int channel1,int channel2) {
    
    for(int i=0;i<channel1;++i){
        for(int j=0;j<row;++j){
            for(int k=0;k<col;++k){
                output[i*row*col+j*row+k]=input1[i*row*col+j*col+k];
            }
        }
    }
    int tmp=channel1*row*col;
    for(int i=0;i<channel2;++i){
        for(int j=0;j<row;++j){
            for(int k=0;k<col;++k){
                output[tmp+i*row*col+j*row+k]=input2[i*row*col+j*col+k];
            }
        }
    }
    
}

void catAVX(float* input1,float* input2, float* output,int row,int col,int channel1,int channel2) {
    
    for(int i=0;i<channel1;++i){
        for(int j=0;j<row;++j){
            for(int k=0;k<col;++k){
                __m256 vec=_mm256_loadu_ps(&input1[i*row*col+j*col+k]);
                _mm256_storeu_ps(&output[i*row*col+j*row+k], vec);
                // output[i*row*col+j*row+k]=input1[i*row*col+j*col+k];
            }
        }
    }
    int tmp=channel1*row*col;
    for(int i=0;i<channel2;++i){
        for(int j=0;j<row;++j){
            for(int k=0;k<col;++k){
                __m256 vec=_mm256_loadu_ps(&input2[i*row*col+j*col+k]);
                _mm256_storeu_ps(&output[tmp+i*row*col+j*row+k], vec);
            }
        }
    }
    
}

int main(){

    int n=64;
    std::vector<float> input1(n*n*n,0);
    std::vector<float> input2(n*n*n,0);

    for(int i=0;i<n*n*n;++i){
        input1[i]=i;
        input2[i]=i;

    }

    std::vector<float> output1(n*n*n*2,0);
    std::vector<float> output2(n*n*n*2,0);
    

    auto time1=measureExecutionTime(cat,input1.data(),input2.data(), output1.data(),n,n,n,n);
    auto time2=measureExecutionTime(catAVX,input1.data(),input2.data(), output1.data(),n,n,n,n);

    // 打印执行时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;  
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;      


    return 0;
}
