# biubiuquik
一个用于学习的算子实现
## 主要使用的实现方式
avx2：256位，处理32位float，一次处理8个数据\
noen：128位，处理32位float，一次处理4个数据\
Ascend：昇腾npu
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

## 依赖库
https://github.com/reyoung/avx_mathfun \
http://gruntthepeon.free.fr/ssemath/neon_mathfun.html