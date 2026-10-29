# 第三阶段、第二阶段、第一阶段与原生 nano-vLLM 性能对照

测试时间：2026-10-29（北京时间）。本次固定实现与参数进行性能测量，未进行性能调优。四方主表使用同一个 Qwen3 Dense 浮点模型，以保证模型架构和目标权重一致；阶段三的量化和 MoE 另作补充，不能把不同模型或不同配置的结果混为四方同条件结果。

## 测试对象与环境

- 原版：本地官方 nano-vLLM `bb823b3e06983d71485a8e1f23715ebd87d98ef8`，其 `nanovllm/` 的 Python 文件与本仓库初始提交 `57ac1cd` 对应文件逐字节一致。本地原版仓库 README 和示例有未提交修改，但被测 Python 包没有修改。
- 第一阶段：独立导出提交 `e0c3da7`，不包含第二、三阶段。
- 第二阶段：独立导出 `c511155`，启用推测解码 `speculative_num_tokens=4`。
- 第三阶段：独立导出 `a1963f0`，为保持与第二阶段同一计算路径，主表同样启用推测解码、K=4，且 `quantization=None`，使用 Qwen3 Dense；不在主表启用 INT4/INT8 或使用 MoE 权重。
- Target 和 Draft 均为同一份本地 Qwen3-0.6B BF16 权重，模型 SHA-256：`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。本机没有现成的更小兼容 Draft，故 Draft 和 Target 同尺寸，而且第二、三阶段每轮从提交序列重建 Draft KV。这不是理想推测配置，不代表使用更小 Draft 的潜在收益。
- 单张 NVIDIA GeForce RTX 5060 8 GiB，TP=1；PyTorch 2.9.1+cu128、Transformers 4.57.3、FlashAttention 2.8.3。桌面进程占用部分显存，GPU 未隔离、未锁定时钟。
- 四组共同参数：`max_num_batched_tokens=512`、`max_num_seqs=8`、`max_model_len=2048`、`gpu_memory_utilization=0.85`、`enforce_eager=False`、`temperature=0.6`、`ignore_eos=True`。第一至三阶段的 `prefill_chunk_size=512` 使用其默认值。普通纯 Decode 可用 CUDA Graph；推测验证走 eager。
- 显存比例、token 预算等是统一预先固定的测试条件，不是搜索出来的最优配置；各进程会根据当时空闲显存自动分配不同数量 KV 块，主表不应被解读为固定 KV 容量下的内核对照。
- 每个引擎启动两个独立进程，各负载先预热 1 轮再测 3 轮，共 6 轮；顺序为二→三→一→原→原→一→三→二。每轮采用一致的随机输入种子，并更换 prompt，降低跨轮 Prefix Cache 命中。随机采样、调度顺序及随机数消耗不同，输出 token 内容无需一致，所有请求的固定输出长度均被检查；本测不比较生成质量。

## 负载与指标口径

1. **短输入批次**：4 个同时提交的请求，每个 128 输入、32 输出 token。
2. **长输入批次**：4 个同时提交的请求，每个 1024 输入、16 输出 token。
3. **错峰混合**：一条 64 输入、64 输出请求先运行，某次 `step()` 结束后其输出数首次达到或超过 4 时，加入两条各为 1024 输入、16 输出的请求。推测路径可一次返回多个 token，因此触发时可能超过 4；这是步边界触发的仿真，不是固定请求到达率的线上测试。

统一外部驱动调用 `add_request()` 和 `step()`，每步返回时观察各请求生成数量并使用同一 CPU 时钟；从首请求提交前开始计时，到所有请求结束并 CUDA 同步。模型初始化、预热和构造 prompt 不计入。吞吐为输出 token 总数 / 墙钟时间；TTFT 为提交到首 token；TPOT 为 `(末 token 时间 - 首 token 时间)/(输出数 - 1)`；ITL 为相邻 token 交付间隔。

表中 TTFT/TPOT 的 P50 先对每轮请求求中位数，ITL P95 先对每轮所有间隔求 P95，最后对 6 轮结果取中位数；首请求最大 ITL 也取 6 轮中位数。每轮只有 3–4 条请求，不能据此推断生产尾延迟。推测路径可能在单个 step 一次交付多个 token：这些 token 的观测时间相同，组内 ITL 为 0，表示驱动的批量交付粒度，并不是每个 token 独立的 GPU 生成时间。测试不含 HTTP、网络或真实异步客户端开销。

## 四方实测与官方理论预期

<!-- BEGIN FOUR WAY TABLE -->
| 场景 | 指标 | 原版 | 第一阶段 | 第二阶段 | 第三阶段 | 三对原版 | 三对一 | 三对二 | 官方理论预期 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 短输入批次 | 输出吞吐 (token/s) | 616.36 | 620.14 | 13.71 | 14.17 | -97.70% | -97.71% | +3.39% | Draft 足够快时推测解码可能提高吞吐；无固定提升率 |
| 短输入批次 | TTFT P50 (ms) | 56.42 | 54.86 | 59.50 | 57.56 | +2.02% | +4.93% | -3.26% | 推测主要作用于 Decode；受 Prefill 调度影响 |
| 短输入批次 | TPOT P50 (ms) | 4.88 | 4.90 | 299.08 | 289.48 | +5833.92% | +5808.35% | -3.21% | 中低负载下推测解码可能降低生成延迟 |
| 短输入批次 | 全请求 ITL P95 (ms) | 5.14 | 5.15 | 1572.79 | 1507.17 | +29197.29% | +29188.68% | -4.17% | 可能改善 ITL；依赖模型、流量与硬件 |
| 短输入批次 | 首请求最大 ITL (ms) | 5.39 | 5.32 | 1617.08 | 1547.56 | +28594.10% | +28979.03% | -4.30% | 可能改善 ITL；无固定比例 |
| 短输入批次 | 目标模型前向调用数 | 32.00 | 32.00 | 27.00 | 27.00 | -15.62% | -15.62% | +0.00% | K=4 时一次验证最多交付 5 token，不含 Draft 开销 |
| 长输入批次 | 输出吞吐 (token/s) | 120.04 | 110.69 | 12.34 | 12.24 | -89.81% | -88.95% | -0.88% | Draft 足够快时推测解码可能提高吞吐；无固定提升率 |
| 长输入批次 | TTFT P50 (ms) | 278.67 | 329.43 | 1891.43 | 1903.39 | +583.03% | +477.78% | +0.63% | 推测主要作用于 Decode；受 Prefill 调度影响 |
| 长输入批次 | TPOT P50 (ms) | 16.99 | 15.52 | 109.45 | 110.14 | +548.07% | +609.77% | +0.63% | 中低负载下推测解码可能降低生成延迟 |
| 长输入批次 | 全请求 ITL P95 (ms) | 11.77 | 55.66 | 786.97 | 775.76 | +6490.58% | +1293.86% | -1.42% | 可能改善 ITL；依赖模型、流量与硬件 |
| 长输入批次 | 首请求最大 ITL (ms) | 339.67 | 56.78 | 466.15 | 463.74 | +36.53% | +716.73% | -0.52% | 可能改善 ITL；无固定比例 |
| 长输入批次 | 目标模型前向调用数 | 23.00 | 24.00 | 21.00 | 21.00 | -8.70% | -12.50% | +0.00% | K=4 时一次验证最多交付 5 token，不含 Draft 开销 |
| 错峰混合 | 输出吞吐 (token/s) | 164.34 | 156.08 | 13.19 | 12.76 | -92.23% | -91.82% | -3.24% | Draft 足够快时推测解码可能提高吞吐；无固定提升率 |
| 错峰混合 | TTFT P50 (ms) | 112.62 | 167.82 | 1228.48 | 1221.29 | +984.45% | +627.75% | -0.58% | 推测主要作用于 Decode；受 Prefill 调度影响 |
| 错峰混合 | TPOT P50 (ms) | 8.39 | 8.90 | 175.22 | 181.01 | +2058.59% | +1934.16% | +3.30% | 中低负载下推测解码可能降低生成延迟 |
| 错峰混合 | 全请求 ITL P95 (ms) | 5.50 | 54.30 | 804.91 | 862.75 | +15575.30% | +1488.92% | +7.19% | 可能改善 ITL；依赖模型、流量与硬件 |
| 错峰混合 | 首请求最大 ITL (ms) | 231.20 | 57.22 | 1117.90 | 1139.24 | +392.76% | +1891.00% | +1.91% | 可能改善 ITL；无固定比例 |
| 错峰混合 | 目标模型前向调用数 | 68.00 | 64.00 | 25.00 | 25.00 | -63.24% | -60.94% | +0.00% | K=4 时一次验证最多交付 5 token，不含 Draft 开销 |
<!-- END FOUR WAY TABLE -->

变化率为 `(第三阶段 / 对照 - 1) × 100%`；吞吐负数表示下降，时延负数表示缩短。官方预期列描述的是相关系统的一般机制，不是针对本硬件或仓库的固定收益。目标前向次数由外部驱动观测普通 `run` 以及每个推测请求的 `run_speculative`：包含 Prefill、排除 Draft 前向。普通 batch 可同时包含多个请求，而本实现推测验证按请求分别执行，因此调用次数不等于执行成本。

<!-- BEGIN FOUR WAY DETAILS -->
- 短输入批次：四组每轮调度步数中位数依次为 32、32、8、8。
  第二阶段：候选接受 576/592（97.30%），每轮验证前向中位数 25。
  第三阶段：候选接受 576/592（97.30%），每轮验证前向中位数 25。
- 长输入批次：四组每轮调度步数中位数依次为 23、24、12、12。
  第二阶段：候选接受 288/288（100.00%），每轮验证前向中位数 12。
  第三阶段：候选接受 288/288（100.00%），每轮验证前向中位数 12。
- 错峰混合：四组每轮调度步数中位数依次为 68、64、14、14。
  第二阶段：候选接受 442/452（97.79%），每轮验证前向中位数 19。
  第三阶段：候选接受 442/452（97.79%），每轮验证前向中位数 19。
<!-- END FOUR WAY DETAILS -->

本次 8 个进程中，原版和第一阶段各自动分配 161 个目标 KV 块；第二、三阶段各分配 93 个，Draft 权重和额外预留占用了可用空间。第二阶段与第三阶段在对应 18 组相同负载/种子下的输出哈希一致；这只是本样本的观测，不等同于随机分布或模型质量的全面验证。二、三阶段之间的几个百分点差异也可能受共享 GPU 和运行顺序影响，不能仅凭这些数据归因于架构注册或称作稳定提升。

## 第三阶段量化功能补充对照

下表单独比较第三阶段的 BF16、INT8、INT4，并关闭推测解码，以隔离量化路径；它们不是主表中的四个同条件对象。使用同一 Qwen3、短输入批次（4×128 输入/32 输出）、温度 0.6、其余配置与主表一致。每种模式各启动一个进程，预热 1 轮并测 3 轮，表中为中位数；轮数较少，结果仅作本次功能性能观察。

<!-- BEGIN QUANT TABLE -->
| 第三阶段模式 | 输出吞吐 (token/s) | 相对 BF16 | TTFT P50 (ms) | TPOT P50 (ms) | 模型张量字节 | 相对 BF16 | 位宽理论提示 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BF16 | 589.96 | +0.00% | 59.08 | 5.04 | 1,213,071,360 | +0.00% | 16-bit 基准 |
| INT8 | 141.52 | -76.01% | 352.56 | 17.82 | 800,997,376 | -33.97% | 仅权重本体约 50%，不含元数据 |
| INT4 | 68.16 | -88.45% | 770.76 | 35.71 | 580,796,416 | -52.12% | 仅权重本体约 25%，不含元数据 |
<!-- END QUANT TABLE -->

`model_tensor_bytes` 按目标模型参数和 buffer 的独立 storage 合计，包括未量化的 embedding、LM head、norm、RoPE、scale/zero；不等于总显存。KV 块数按可用显存自动变化，不能以此推断固定 KV 配置下的峰值节省。权重量化在加载浮点检查点后进行，因此初始化仍需浮点权重和临时空间；不同量化模式的随机输出哈希在本次三轮中均不同，但随机采样下哈希不同不能单独衡量量化误差或模型质量。

## 官方理论及适用范围

- [vLLM Chunked Prefill 文档](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)描述 Prefill 分块和 Decode 优先可能改善 ITL，并平衡计算和带宽；具体效果依赖负载和预算。
- [vLLM Speculative Decoding 文档](https://docs.vllm.ai/en/latest/features/speculative_decoding/)指出该技术适用于中低请求率下可能改善延迟的场景，实际收益依赖模型、请求流、硬件和采样。它没有为本项目或本次硬件提供固定提升百分比。
- [推测解码论文](https://proceedings.mlr.press/v202/leviathan23a.html)在特定 T5-XXL 实验报告 2–3 倍加速；这是论文的特定实测，不是本项目 Qwen3、同尺寸 Draft 的承诺。理想化独立接受率 α、K=4 时，每次验证期望输出为 `1 + α + α² + α³ + α⁴`；这不包括 Draft、KV 重建、验证、调度与采样成本，不能作为端到端加速率。
- [vLLM 量化文档](https://docs.vllm.ai/en/stable/features/quantization/)介绍低精度方案；仅按权重位宽粗算，INT8 和 INT4 权重本体分别为 16-bit 的 1/2 和 1/4。scale、zero、未量化层、KV 与临时内存会改变整体结果；低位宽不保证吞吐提升或输出质量一致。
- MoE 将 token 路由到少量专家，但所有专家权重仍占显存。第三阶段本地 MoE 为单卡 PyTorch 参考后端，不应将不同 checkpoint 的吞吐当作同模型的架构加速。

## MoE 独立补充测试

第三阶段还使用本地 Huihui-Qwen3-MoE-1B-A0.6B，关闭推测与量化，在与量化补充相同的 4×128 输入/32 输出短批次、预热一轮后测量 3 轮。输出吞吐中位数 **42.53 token/s**，TTFT P50 **114.38 ms**，TPOT P50 **93.37 ms**，全请求 ITL P95 **104.46 ms**；模型张量占用 **2,270,208,000 字节**、自动分配 **118 个 KV 块**。启动参数的 `enforce_eager=False` 在该 MoE 后端被引擎自动覆盖为 eager，运行中不使用 CUDA Graph。

该 MoE checkpoint 的权重 SHA-256 为 `e893bd29c30948c66b7bb6da4ede05684aa35eea99c8b2a07bfd8a9aa56e1b17`，原始数据为 `data/feature-moe.json`。它与四方主表采用不同模型权重和计算量，不能直接计算相对提升；此前另一组 MoE 单请求与专家分配记录见 [阶段三功能基准](../../../benchmarks/STAGE3_RESULTS.md)。真实第三方 AWQ/GPTQ checkpoint、TP>1、不同大小 Draft 和生产在线负载尚未在本次验证。

## 复现

```bash
cd /home/ubuntu/ai-infra/minivllm/myvllm/wwvllm
mkdir -p /tmp/wwvllm-fourway-20261029/{stage1,stage2,stage3}/wwvllm
git archive e0c3da7 | tar -x -C /tmp/wwvllm-fourway-20261029/stage1/wwvllm
git archive c511155 | tar -x -C /tmp/wwvllm-fourway-20261029/stage2/wwvllm
git archive a1963f0 | tar -x -C /tmp/wwvllm-fourway-20261029/stage3/wwvllm
find /tmp/wwvllm-fourway-20261029 -type d -name __pycache__ -prune -exec rm -r {} +
PY=/home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python
SCRIPT='docs/性能对照/阶段三/scripts/four_way_compare.py'
PYTHONPATH=/home/ubuntu/ai-infra/minivllm/nano-vllm PYTHONDONTWRITEBYTECODE=1 "$PY" "$SCRIPT" --engine nano --output /tmp/nano.json
PYTHONPATH=/tmp/wwvllm-fourway-20261029/stage1 PYTHONDONTWRITEBYTECODE=1 "$PY" "$SCRIPT" --engine stage1 --output /tmp/stage1.json
PYTHONPATH=/tmp/wwvllm-fourway-20261029/stage2 PYTHONDONTWRITEBYTECODE=1 "$PY" "$SCRIPT" --engine stage2 --output /tmp/stage2.json
PYTHONPATH=/tmp/wwvllm-fourway-20261029/stage3 PYTHONDONTWRITEBYTECODE=1 "$PY" "$SCRIPT" --engine stage3 --output /tmp/stage3.json
# 量化补充：分别将 --mode 设置为 bf16、int8、int4
PYTHONPATH=/tmp/wwvllm-fourway-20261029/stage3 PYTHONDONTWRITEBYTECODE=1 "$PY" 'docs/性能对照/阶段三/scripts/quantization_compare.py' --mode bf16 --output /tmp/bf16.json
# MoE 独立补充（同脚本、不同模型；引擎自动使用 eager）
PYTHONPATH=/tmp/wwvllm-fourway-20261029/stage3 PYTHONDONTWRITEBYTECODE=1 "$PY" 'docs/性能对照/阶段三/scripts/quantization_compare.py' --mode bf16 --model /home/ubuntu/ai-infra/minivllm/Models/Huihui-MoE-1B-A0.6B --output /tmp/moe.json
```

四方按上文对称顺序各执行两次并保存不同文件；`data/` 保存本次逐轮结果和相对 token 交付时间。表格由 `scripts/summarize.py` 从原始数据生成；不修改任何被测引擎代码。
