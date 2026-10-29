# 阶段三实验 01：量化 GEMM 的大批次行分块

## 数据、假设与设计

此前[阶段三对照](../../../性能对照/阶段三/README.md)在 4×128→32 下测得 BF16 589.96、INT8 141.52、INT4 68.16 输出 token/s；量化节省权重显存，但吞吐明显低于 BF16。量化 `Linear` 原先无论矩阵行数 M 都使用 Triton `BM=16、BN=32、BK=32`，量化权重、分组 scale 和 zero 对每 16 行重复加载。假设：较大的 prefill 批次增加 BM 可复用权重和分组数据；小 M 的 Decode 若同时增大 BM，可能增加无效计算。

先用 CUDA event 计时扫描 BM/BN/BK，热身 100 ms、测量 300 ms，测试 BF16 输入、每组 128、对称量化，模型相关的 K=1024、N=1024/3072 形状；完整数据为 `data/tiles.json`。以下为其中两种 M=128 的代表结果（毫秒，越低越好）：

| bits | M×K × N | 原分块 16×32×32 | 候选 32×32×32 | 延迟变化 |
|---|---|---:|---:|---:|
| INT8 | 128×1024 × 1024 | 0.2219 | 0.1305 | -41.2% |
| INT8 | 128×1024 × 3072 | 0.6232 | 0.3362 | -46.1% |
| INT4 | 128×1024 × 1024 | 0.4885 | 0.2731 | -44.1% |
| INT4 | 128×1024 × 3072 | 1.3921 | 0.7001 | -49.7% |

修改 `quantization/linear.py`：M≥64 时 BM=32，其他情况仍 BM=16；BN=32、BK=32、计算公式、量化格式、任意 `g_idx`、CPU 回退、bias 和输出 dtype 不变。补充测量的 M=64 与 M=512、K=1024/3072、N=1024 在 INT4/INT8 下均降低约 35.3%–48.2% 的算子耗时，见 `data/boundary.json`。64 仍是启发式阈值，并非经过穷举的最优阈值；目前没有证明所有硬件及 M、N、K 都受益。MoE 专家路由未调整。

## 端到端对照

基线源码固定 `afe48dd`，候选为本次修改；相同 RTX 5060 8 GiB、PyTorch 2.9.1+cu128、Triton 3.5.1、Qwen3-0.6B，单卡、推测关闭、temperature=0.6、4 个 128 token 输入各生成 32 token，进程独立，每种预热一次后重复两次。先测两个基线再测两个候选，并非交错 ABBA；共享 GPU 和少量重复影响结论。吞吐为输出 token 总数/端到端总时间，其改善包含 prefill 对总时间的影响，不能解释为 Decode 内核同幅加速。

| 模式 | 输出吞吐 token/s 基线→候选 | 变化 | TTFT p50 秒 基线→候选 | TPOT p50 秒 基线→候选 | ITL p95 秒 基线→候选 |
|---|---:|---:|---:|---:|---:|
| INT8 | 139.01→169.54 | +21.96% | 0.36253→0.19399 (-46.49%) | 0.018006→0.018094 (+0.49%) | 0.018689→0.018878 (+1.01%) |
| INT4 | 66.27→83.41 | +25.86% | 0.79593→0.40651 (-48.93%) | 0.036630→0.036389 (-0.66%) | 0.037282→0.037004 (-0.75%) |

INT8 的 TPOT 和 ITL 小幅退化，幅度较小且本轮未控制进程顺序，不能断言原因。两个模式分别两轮的输出 token 哈希与基线相同；这只是有限样本，不是对所有采样结果的证明。PyTorch 报告的最大已分配显存 INT8 均为 5,772,832,768 bytes，INT4 均为 5,574,651,904 bytes。权重格式不变；对照先前 BF16 结果仍可见较大差距，但它来自先前测试，不能当作本轮配对比较。

## 验证与复现

`tests/test_stage3.py` 共 15 项通过，新增覆盖 M=4/63/64/129、奇数 K、FP16/BF16、INT4/INT8、非连续 `g_idx` 和非对称 zero，与反量化参考比较。调度、推测及异步的 CPU 测试 13 项通过。真实及合成模型集成回归 7 项通过，覆盖 Llama、Qwen3-MoE、AWQ/GPTQ、量化目标与推测解码联动、真实 MoE checkpoint 和普通推测解码。改动涉及 `quantization/linear.py`、`tests/test_stage3.py` 及本目录报告、脚本和原始 JSON。

在仓库根目录使用 `/home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python` 和 `PYTHONPATH=..`：算子测量运行 `scripts/tile_bench.py --output ...`；端到端使用 `docs/性能对照/阶段三/scripts/quantization_compare.py --mode int8|int4 --repeats 2 --output ...`。基线应以 `git archive afe48dd` 导出包后通过 `PYTHONPATH` 指向该副本。数据保存了各轮时延、哈希、目标前向次数和显存指标。

## 代码变化

原策略始终按 16 行启动每个 tile：

```python
_quant_gemm[(triton.cdiv(x.shape[0], 16), triton.cdiv(self.out_features, 32))](
    ..., self.bits == 4, 16, 32, 32)
```

修改后由运行时行数选择分块，核函数及反量化的数学定义不变：

```python
block_m = 32 if x.shape[0] >= 64 else 16
_quant_gemm[(triton.cdiv(x.shape[0], block_m), triton.cdiv(self.out_features, 32))](
    ..., self.bits == 4, block_m, 32, 32)
```
