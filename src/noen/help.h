#ifndef HELLP
#define HELLP
#include <iostream>
#include <chrono>
#include <functional>

template<typename Func, typename... Args>
double measureExecutionTime(Func&& func, Args&&... args) {

    auto start = std::chrono::high_resolution_clock::now();


    std::forward<Func>(func)(std::forward<Args>(args)...);


    auto end = std::chrono::high_resolution_clock::now();


    std::chrono::duration<double> elapsed = end - start;

    return elapsed.count();
}


inline void check(float* output1,float* output2,int n){
    for (int i = 0; i < n; ++i) {
        if(output1[i]-output2[i]>0.001f||output2[i]-output1[i]>0.001f){
            float tmp=output1[i]-output2[i];
            printf("%f,%d,fault\n",tmp,i);
            return ;
        }        
    }
    printf("ok\n");
}

#endif