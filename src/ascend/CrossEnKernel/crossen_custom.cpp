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

constexpr int32_t TOTAL_LENGTH = 8 * 512;                            // total length of data
constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         // length computed of each core
constexpr int32_t TILE_NUM = 8;                                       // split data into 8 tiles for each core
constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; // separate to 2 parts, due to double buffer

class KernelCrossEn {
public:
    __aicore__ inline KernelCrossEn() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half *)y + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ half *)z + BLOCK_LENGTH * AscendC::GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(calcBuf1, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(calcBuf2, TILE_LENGTH * sizeof(half));
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
        AscendC::LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        AscendC::LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        AscendC::LocalTensor<half> tmp1=calcBuf1.Get<half>();
        AscendC::LocalTensor<half> tmp2=calcBuf2.Get<half>();
        AscendC::LocalTensor<half> tmp3=calcBuf3.Get<half>();
        AscendC::LocalTensor<half> one=calcBuf4.Get<half>();
        half val=1.0f;
        AscendC::Duplicate<half>(one, val, TILE_LENGTH);
        AscendC::Ln(tmp1,yLocal,TILE_LENGTH);
        AscendC::Mul(tmp1,tmp1,xLocal,TILE_LENGTH);
        AscendC::Sub(tmp2,one,yLocal,TILE_LENGTH);
        AscendC::Ln(tmp2,tmp2,TILE_LENGTH);
        AscendC::Sub(tmp3,one,xLocal,TILE_LENGTH);
        AscendC::Mul(tmp2,tmp2,tmp3,TILE_LENGTH);
        AscendC::Add(zLocal, tmp1, tmp2, TILE_LENGTH);

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
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf1; 
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf2;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf3;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf4;
    AscendC::GlobalTensor<half> xGm;
    AscendC::GlobalTensor<half> yGm;
    AscendC::GlobalTensor<half> zGm;
};

extern "C" __global__ __aicore__ void crossen_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z)
{
    KernelCrossEn op;
    op.Init(x, y, z);
    op.Process();
}
