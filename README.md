# biubiuquik
一个用于学习的算子实现
## 主要使用的实现方式
avx2：256位，处理32位float，一次处理8个数据\
noen：128位，处理32位float，一次处理4个数据\
Ascend：昇腾npu 910A，主要计算half数据类型，使用基础api实现\
CUDA：nvidia显卡\
Triton：GPU算子编译器
## 主要实现的算子
Linear
ReLU
Sigmoid
BatchNorm
ELU
HardSigmoid
HardWish
KL散度
MaxPooling
MeanSquaredError
PReLU
SiLU
Softmax
Tanh
## 编译指令
axv2
```shell
cd src/avx2
gcc -mavx2 -g -o linearv linear.cpp  -lstdc++ -lm
```
neon 32位
```shell
cd src/neon
arm-linux-gnueabihf-gcc   --static -mfpu=neon -mfloat-abi=hard -std=c++11  -o linear linear.cpp -lstdc++ -lm
```

AscendC
```shell
cd src/ascend/CrossEnKernel
./bash run.sh -r npu -v Ascend910A
```

CUDA
```shell
cd src/cuda
nvcc -o silu silu.cu
```

## 主要问题
avx：exp和log函数精度不足，计算不够快，以及向量进行累加以后精度也会不足\
noen：exp和log函数精度不足，计算不够快\
AscendC：exp和ln函数精度不够\
CUDA：由于要从内存拷贝到显存，非计算密集的算子实现不如cpu实现块，可以用shared memory加快计算

## 依赖库
https://github.com/reyoung/avx_mathfun \
http://gruntthepeon.free.fr/ssemath/neon_mathfun.html