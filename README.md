# TMA Depthwise Conv2D (NHWC) on Blackwell B200

针对 NVIDIA Blackwell B200 (sm_100) 编写的 **NHWC Depthwise Conv2D 前向** 实现，用于学习 **TMA (Tensor Memory Accelerator)**、`mbarrier` 以及 **warp-specialized 多级流水** 这些特性。

C++/CUDA 只负责 kernel；正确性校验和性能测试都用 PyTorch CUDA Extension 在 Python 侧驱动。

## 版本

| id | 说明 |
| -- | ---- |
| v0 | naive kernel，不走 smem、不走 TMA，用 `float4` 向量化读写 gmem |
| ws | TMA + warp specialization + 多级流水，`n_stage` 可配（支持 2/3/4） |

固定配置：
- fp16 输入 / fp32 累加 / fp16 输出
- `R = S = 3`, `stride = 1`, `padding` 任意（测试用 1）
- `BlockP = 8`, `BlockQ = 32`, `BlockC = 64`

## 安装


### 拉取仓库（含 submodule）

```bash
git clone <repo-url>
cd tma-depthwise-conv
git submodule update --init --recursive   # 拉 third_party/cutlass
```

或直接一步到位：`git clone --recursive <repo-url>`。

### 安装扩展

```bash
pip install -e . --no-build-isolation
```
## 单元测试

```bash
pytest tests/
```

## 性能测试

```bash
python -m tests.bench
```

运行结束后会在 `tests/bench.png` 生成四条曲线（cuDNN / v0 / ws n_stage=2 / ws n_stage=4）的耗时对比图。

## Producer / Consumer 等待时间分析

`ws` kernel 内部的 `cute::wait_barrier` 调用埋了 `clock64()` 探针（通过模板 bool `kStallProbe` 控制，关闭时整段代码被 `if constexpr` 编译消除，对正常 kernel 零开销）。开启探针的入口是 `dwconv_ws_probe`，每个 CTA 写回 3 列 cycle 计数：

| 列 | 含义 | 大代表什么 |
|----|------|----------|
| 0 | producer 在 `empty_bar` 上等待的 cycles | consumer 慢 → **consumer 瓶颈** |
| 1 | consumer 在 `full_bar` 上等待的 cycles，4 个 consumer warp 求和 | producer 慢 → **producer 瓶颈** |
| 2 | warp 0 lane 0 测得的整个 kernel 总 cycles | 归一化基准 |

直接跑探针测试（脚本会跨 `n_stage ∈ {2, 3, 4}` 和若干 shape 汇总打印一行 / 用例）：

```bash
python tests/test_ws_time.py
# 或作为 pytest 用例（每条用例打印一行）：
pytest -s tests/test_ws_time.py
```

结果：
```text
root@b200:/tma-depthwise-conv# CUDA_VISIBLE_DEVICES=7 python tests/test_ws_time.py
================================================================================
Warp-specialized depthwise conv: producer/consumer stall probe
  CONSUMER_WARPS = 4 (consumer stall column is summed across warps; divide by 4 for per-warp comparison with producer)
  warmup = 2, measured launches accumulated = 5
================================================================================
  N=  8 H= 56 W= 56 C= 256  stage=2  blks=  224  prod μ=         0  p95=         0  cons/warp μ=      7920  p95=     10231  prod_tot μ=      25824  cons_tot/warp μ=     161649  prod/prod_tot=  0.0%  cons/cons_tot=  4.9%  -> producer bottleneck (prod/cons=0.00)
  N=  8 H= 56 W= 56 C= 256  stage=3  blks=  224  prod μ=         0  p95=         0  cons/warp μ=      6292  p95=      8081  prod_tot μ=      23550  cons_tot/warp μ=     140454  prod/prod_tot=  0.0%  cons/cons_tot=  4.5%  -> producer bottleneck (prod/cons=0.00)
  N=  8 H= 56 W= 56 C= 256  stage=4  blks=  224  prod μ=         0  p95=         0  cons/warp μ=      6169  p95=      7945  prod_tot μ=      24256  cons_tot/warp μ=     141457  prod/prod_tot=  0.0%  cons/cons_tot=  4.3%  -> producer bottleneck (prod/cons=0.00)

  N= 16 H=112 W=224 C= 128  stage=2  blks=  448  prod μ=    372353  p95=    382289  cons/warp μ=     24599  p95=     30192  prod_tot μ=     412816  cons_tot/warp μ=     549249  prod/prod_tot= 90.2%  cons/cons_tot=  4.5%  -> consumer bottleneck (prod/cons=15.14)
  N= 16 H=112 W=224 C= 128  stage=3  blks=  448  prod μ=    243452  p95=    248017  cons/warp μ=     18621  p95=     22896  prod_tot μ=     269195  cons_tot/warp μ=     442279  prod/prod_tot= 90.6%  cons/cons_tot=  4.2%  -> consumer bottleneck (prod/cons=13.07)
  N= 16 H=112 W=224 C= 128  stage=4  blks=  448  prod μ=    187592  p95=    195149  cons/warp μ=     21103  p95=     28805  prod_tot μ=     216824  cons_tot/warp μ=     447111  prod/prod_tot= 86.6%  cons/cons_tot=  4.7%  -> consumer bottleneck (prod/cons=8.89)

  N= 32 H=224 W=336 C= 256  stage=2  blks= 3584  prod μ=    654472  p95=    665517  cons/warp μ=     22419  p95=     31504  prod_tot μ=     691117  cons_tot/warp μ=     823552  prod/prod_tot= 94.7%  cons/cons_tot=  2.7%  -> consumer bottleneck (prod/cons=29.19)
  N= 32 H=224 W=336 C= 256  stage=3  blks= 3584  prod μ=    471076  p95=    473957  cons/warp μ=     17648  p95=     20466  prod_tot μ=     498762  cons_tot/warp μ=     668155  prod/prod_tot= 94.5%  cons/cons_tot=  2.6%  -> consumer bottleneck (prod/cons=26.69)
  N= 32 H=224 W=336 C= 256  stage=4  blks= 3584  prod μ=    413064  p95=    417515  cons/warp μ=     19202  p95=     23687  prod_tot μ=     448505  cons_tot/warp μ=     676192  prod/prod_tot= 92.1%  cons/cons_tot=  2.8%  -> consumer bottleneck (prod/cons=21.51)

  N= 64 H=224 W=336 C= 256  stage=2  blks= 7168  prod μ=    653691  p95=    662886  cons/warp μ=     21203  p95=     26836  prod_tot μ=     689618  cons_tot/warp μ=     822160  prod/prod_tot= 94.8%  cons/cons_tot=  2.6%  -> consumer bottleneck (prod/cons=30.83)
  N= 64 H=224 W=336 C= 256  stage=3  blks= 7168  prod μ=    470633  p95=    473317  cons/warp μ=     17097  p95=     19322  prod_tot μ=     497813  cons_tot/warp μ=     667268  prod/prod_tot= 94.6%  cons/cons_tot=  2.6%  -> consumer bottleneck (prod/cons=27.53)
  N= 64 H=224 W=336 C= 256  stage=4  blks= 7168  prod μ=    411790  p95=    415646  cons/warp μ=     17921  p95=     21756  prod_tot μ=     445510  cons_tot/warp μ=     673240  prod/prod_tot= 92.4%  cons/cons_tot=  2.7%  -> consumer bottleneck (prod/cons=22.98)

```

结论：

- **小 shape**（N=8, 56×56, C=256）：producer 单 round 就跑完整个 tile 序列，根本没机会等 `empty_bar`，所以 `prod_stall = 0`。这种规模下探针不具参考性，结果里 "producer bottleneck" 是除零得出的伪标签。
- **大 shape**（112×224 / 224×336）：producer stall 稳定占 kernel 总时间 **85%–95%**，consumer stall（除以 4 个 warp 后）只占 3%–5%，确认 **consumer 的 FMA 是瓶颈**，TMA 数据搬运绝大部分时间在空转等 buffer 释放。
- **加 stage 的边际收益**：`n_stage` 从 2 → 4，prod_stall 占比从 ~94% 降到 ~92%，提升非常有限。说明问题不在流水深度，而在 consumer 单位时间能消化的数据量。后续优化方向应是 consumer 计算路径（hfma2 / mma 重写、增加 consumer warp 数），而不是继续加 stage 或优化 TMA 路径。

### 测试环境

- GPU: NVIDIA B200 (sm_100)
- Driver: 580.95.05
- CUDA: 13.0 (driver) / 12.9 (nvcc toolkit)
- PyTorch: 2.9.1+cu129
- cuDNN: 9.10.2

![benchmark](tests/bench.png)
