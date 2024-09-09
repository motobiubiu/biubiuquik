/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

constexpr int32_t TOTAL_LENGTH = 8 * 2048;                            // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // separate to 2 parts, due to double buffer

class KernelElu {
public:
    __aicore__ inline KernelElu() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        // yGm.SetGlobalBuffer((__gm__ half *)y , sizeof(half));
        zGm.SetGlobalBuffer((__gm__ half *)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        // pipe.InitBuffer(inQueueY, BUFFER_NUM,  sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        param=*((__gm__ half *)y);
        pipe.InitBuffer(calcBuf1, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(calcBuf2, TILE_LENGTH * sizeof(int8_t));
        pipe.InitBuffer(calcBuf3, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(calcBuf4, TILE_LENGTH * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        // AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        // AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        // inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        // AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        // AscendC::Add(zLocal, xLocal, yLocal, TILE_LENGTH);
        AscendC::LocalTensor<half> zero=calcBuf1.Get<half>();
        AscendC::LocalTensor<int8_t> maskgre=calcBuf2.Get<int8_t>();
        AscendC::LocalTensor<half> exp=calcBuf3.Get<half>();
        AscendC::LocalTensor<half> params=calcBuf4.Get<half>();

        AscendC::Duplicate<half>(zero, 0.f, TILE_LENGTH);
        AscendC::Compare(maskgre,xLocal,zero,AscendC::CMPMODE::GT,TILE_LENGTH);

        AscendC::Duplicate<half>(zero, 1.f, TILE_LENGTH);
        AscendC::Sub(exp,xLocal,zero,TILE_LENGTH);
        AscendC::Exp(exp,exp,TILE_LENGTH);
        AscendC::Duplicate<half>(params, param, TILE_LENGTH);
        AscendC::Mul(exp,exp,params,TILE_LENGTH);
        // AscendC::Add(zLocal,xLocal,zero,TILE_LENGTH);

        AscendC::Select(zLocal, maskgre, xLocal, exp, AscendC::SELMODE::VSEL_CMPMASK_SPR, TILE_LENGTH);        

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        // inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        AscendC::DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    half param;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf1;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf3;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf4;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
};

extern "C" __global__ __aicore__ void elu_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KernelElu op;
    op.Init(x, y, z);
    op.Process();
}
