#include <vector>
#include <cmath>
#include <iostream>
#include <immintrin.h>
#include "help.h"
#include "fun.h"

__m256 exp256_ps1(__m256 x) {
/* Modified code from this source: https://github.com/reyoung/avx_mathfun

   AVX implementation of exp
   Based on "sse_mathfun.h", by Julien Pommier
   http://gruntthepeon.free.fr/ssemath/
   Copyright (C) 2012 Giovanni Garberoglio
   Interdisciplinary Laboratory for Computational Science (LISC)
   Fondazione Bruno Kessler and University of Trento
   via Sommarive, 18
   I-38123 Trento (Italy)
  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.
  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:
  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.
  (this is the zlib license)

*/
/* 
  To increase the compatibility across different compilers the original code is
  converted to plain AVX2 intrinsics code without ingenious macro's,
  gcc style alignment attributes etc.
  Moreover, the part "express exp(x) as exp(g + n*log(2))" has been significantly simplified.
  This modified code is not thoroughly tested!
*/


__m256   exp_hi        = _mm256_set1_ps(88.3762626647949f);
__m256   exp_lo        = _mm256_set1_ps(-88.3762626647949f);

__m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f);
__m256   inv_LOG2EF    = _mm256_set1_ps(0.693147180559945f);

__m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4);
__m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3);
__m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3);
__m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2);
__m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1);
__m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1);
__m256   fx;
__m256i  imm0;
__m256   one           = _mm256_set1_ps(1.0f);

        x     = _mm256_min_ps(x, exp_hi);
        x     = _mm256_max_ps(x, exp_lo);

  /* express exp(x) as exp(g + n*log(2)) */
        fx     = _mm256_mul_ps(x, cephes_LOG2EF);
        fx     = _mm256_round_ps(fx, _MM_FROUND_TO_NEAREST_INT |_MM_FROUND_NO_EXC);
__m256  z      = _mm256_mul_ps(fx, inv_LOG2EF);
        x      = _mm256_sub_ps(x, z);
        z      = _mm256_mul_ps(x,x);

__m256  y      = cephes_exp_p0;
        y      = _mm256_mul_ps(y, x);
        y      = _mm256_add_ps(y, cephes_exp_p1);
        y      = _mm256_mul_ps(y, x);
        y      = _mm256_add_ps(y, cephes_exp_p2);
        y      = _mm256_mul_ps(y, x);
        y      = _mm256_add_ps(y, cephes_exp_p3);
        y      = _mm256_mul_ps(y, x);
        y      = _mm256_add_ps(y, cephes_exp_p4);
        y      = _mm256_mul_ps(y, x);
        y      = _mm256_add_ps(y, cephes_exp_p5);
        y      = _mm256_mul_ps(y, z);
        y      = _mm256_add_ps(y, x);
        y      = _mm256_add_ps(y, one);

  /* build 2^n */
        imm0   = _mm256_cvttps_epi32(fx);
        imm0   = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
        imm0   = _mm256_slli_epi32(imm0, 23);
__m256  pow2n  = _mm256_castsi256_ps(imm0);
        y      = _mm256_mul_ps(y, pow2n);
        return y;
}


void elu(const std::vector<float>& input, std::vector<float>& output,float alpha) {
    
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = input[i]>0?input[i]:alpha*(std::exp(input[i])-1);
    }
    
}

void eluAVX(const float* input, float* output,int n,float alpha) {
    
    for (int i = 0; i <= n-8; i+=8) {
        __m256 vec_zero=_mm256_set1_ps(0.0f);
        __m256 vec=_mm256_loadu_ps(&input[i]);
        __m256 vec_exp=exp256_ps1(vec);

        __m256 vec_maskless=_mm256_cmp_ps(vec, vec_zero, _CMP_LT_OQ);
        __m256 vec_res=_mm256_blendv_ps(vec,_mm256_mul_ps(_mm256_set1_ps(alpha), _mm256_sub_ps(vec_exp, _mm256_set1_ps(1.0f))), vec_maskless);

        _mm256_storeu_ps(&output[i], vec_res);
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
    float alpha=0.5f;
    
    auto time1=measureExecutionTime(elu,input,output1,alpha);
    auto time2=measureExecutionTime(eluAVX,input.data(),output2.data(),n,alpha);

    // 打印时间
    std::cout << "Elapsed time: " << time1 << " seconds" << std::endl;     
    std::cout << "Elapsed time: " << time2 << " seconds" << std::endl;     

    check(output1.data(), output2.data(), n);


    return 0;
}
