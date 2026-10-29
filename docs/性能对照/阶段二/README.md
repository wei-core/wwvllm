# 第二阶段、第一阶段与原生 nano-vLLM 性能对照

测试时间：2026-10-28（北京时间）。本报告为固定实现、固定配置的性能测量，没有对引擎进行性能调优。三方使用相同模型、请求驱动和采样参数；第二阶段开启推测解码。以下结果来自本次独立重跑，不与上一份显存利用率不同的[阶段一对照](../阶段一/README.md)混合统计。

## 对象与条件

- 原版 nano-vLLM：`bb823b3e06983d71485a8e1f23715ebd87d98ef8`；本地 `nanovllm` 全部 Python 源文件与 wwvllm 初始提交 `57ac1cd` 对应文件一致。
- 第一阶段：从 `e0c3da7` 导出的 wwvllm，未包含后续阶段。
- 第二阶段：从 `c511155` 导出的 wwvllm，启用 `enable_speculative=True`、`speculative_num_tokens=4`。本机没有现成的更小且 tokenizer 兼容的 Qwen3 Draft，故 Draft 与 Target 都是同一份 Qwen3-0.6B 权重；Draft 不比 Target 小，且每轮从已提交序列重建 KV。这是明确限制，不代表理想草稿配置。
- 硬件/软件：单张 GeForce RTX 5060 8 GiB，PyTorch 2.9.1+cu128、Transformers 4.57.3、FlashAttention 2.8.3、Qwen3-0.6B BF16，TP=1。GPU 共享桌面进程，没有锁定频率或隔离负载。
- 三组均为 `max_num_batched_tokens=512`、`max_num_seqs=8`、`max_model_len=2048`、`gpu_memory_utilization=0.85`、CUDA Graph 开启（`enforce_eager=False`）、`temperature=0.6`、忽略 EOS、固定输出长度；阶段一和二的 Prefill chunk 为默认 512。第二阶段的推测验证使用 eager，纯普通 Decode 仍可使用 Graph。显存比例从上一份报告的 0.60 改为 0.85，是为同卡加载 Target + Draft 及 KV 留出空间的统一测试条件，不是调参搜索；KV 自动分配量和草稿权重开销仍然不同。
- 模型文件 `model.safetensors` SHA-256：`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
- 顺序：第二阶段 → 第一阶段 → 原版 → 原版 → 第一阶段 → 第二阶段；每次独立启动进程，每种负载预热 1 轮后测量 3 轮，因此每引擎每负载 6 轮。每轮 prompt 使用相同随机种子生成，轮间更换以避免跨轮复用 Prefix Cache。

## 工作负载与测量口径

1. **短输入批次**：同时提交 4 个请求，每个 128 输入、32 输出 token。
2. **长输入批次**：同时提交 4 个请求，每个 1024 输入、16 输出 token。
3. **错峰混合**：先提交一个 64 输入、64 输出请求；当它在一次 `step()` 结束时已生成至少 4 个 token，再提交两个 1024 输入、16 输出请求。第二阶段可能一个 step 返回多个 token，触发时已输出数量可以超过 4，属于以步骤边界为触发的负载，非固定到达时间或真实网络在线压测。

外部驱动调用各版本 `add_request()` / `step()`，每个 step 完成后观察各请求生成的 token 数并打同一 CPU 时钟；预热后，从首次提交之前到全部请求完成且 CUDA 同步为墙钟时间。输出吞吐为输出 token 数除以该时间；TTFT 为提交到首次 token；每个请求 TPOT 为 `(末 token - 首 token)/(输出数-1)`；ITL 为相邻 token 交付间隔。表中的 P50/P95 先按单轮请求或 token 样本计算，再取 6 轮中位数。首请求最大 ITL 为每轮首请求最大间隔，再取 6 轮中位数。每轮仅有 3–4 个请求，不能用来推断生产 P95/P99。

第二阶段一个验证 step 可以同时返回多个 token，驱动只能观测 step 结束时间：同一批 token 被赋予相同时间，因此其组内 ITL 记为 0；这反映该驱动的批量交付粒度，不是独立 token 的 GPU 生成时间。第二阶段每个 step 还可能对多个请求分别串行执行验证，因此 step 等待包括同批其他请求的工作。原版没有相同的异步 API，测试不包含 HTTP、网络和实际客户端并发提交；随机采样、调度方式及随机数消耗可能使三方输出 token 不同，本测试只检查预期输出数量，不是精度或质量比较。

## 三方实测与官方理论预期

<!-- BEGIN GENERATED TABLE -->
| 场景 | 指标 | 原版 nano-vLLM | 第一阶段 | 第二阶段 | 二阶段对一阶段 | 二阶段对原版 | 官方理论预期 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 短输入批次 | 输出吞吐 (token/s) | 624.33 | 599.47 | 13.67 | -97.72% | -97.81% | Draft 足够快时可能提高吞吐；无固定比例 |
| 短输入批次 | TTFT P50 (ms) | 54.04 | 59.65 | 55.55 | -6.87% | +2.80% | 推测主要作用于 Decode；TTFT 取决于调度 |
| 短输入批次 | TPOT P50 (ms) | 4.86 | 4.96 | 300.23 | +5946.94% | +6077.99% | 中低负载下可能降低生成延迟；无固定比例 |
| 短输入批次 | 全请求 ITL P95 (ms) | 5.43 | 5.57 | 1540.05 | +27566.83% | +28276.97% | 可能改善 ITL；依赖模型、流量与硬件 |
| 短输入批次 | 首请求最大 ITL (ms) | 5.63 | 5.68 | 1660.86 | +29161.54% | +29409.09% | 可能改善 ITL；无固定比例 |
| 短输入批次 | 目标模型前向调用数 | 32.00 | 32.00 | 27.00 | -15.62% | -15.62% | 一次验证最多产生 K+1 token；不含 Draft 开销 |
| 长输入批次 | 输出吞吐 (token/s) | 122.46 | 98.10 | 11.65 | -88.12% | -90.49% | Draft 足够快时可能提高吞吐；无固定比例 |
| 长输入批次 | TTFT P50 (ms) | 274.38 | 367.45 | 1892.75 | +415.11% | +589.82% | 推测主要作用于 Decode；TTFT 取决于调度 |
| 长输入批次 | TPOT P50 (ms) | 16.47 | 17.52 | 117.04 | +568.14% | +610.51% | 中低负载下可能降低生成延迟；无固定比例 |
| 长输入批次 | 全请求 ITL P95 (ms) | 11.51 | 67.84 | 799.74 | +1078.86% | +6850.72% | 可能改善 ITL；依赖模型、流量与硬件 |
| 长输入批次 | 首请求最大 ITL (ms) | 333.35 | 72.46 | 466.76 | +544.16% | +40.02% | 可能改善 ITL；无固定比例 |
| 长输入批次 | 目标模型前向调用数 | 23.00 | 24.00 | 21.00 | -12.50% | -8.70% | 一次验证最多产生 K+1 token；不含 Draft 开销 |
| 错峰混合 | 输出吞吐 (token/s) | 168.47 | 151.89 | 13.10 | -91.38% | -92.23% | Draft 足够快时可能提高吞吐；无固定比例 |
| 错峰混合 | TTFT P50 (ms) | 107.47 | 178.07 | 1361.69 | +664.68% | +1167.02% | 推测主要作用于 Decode；TTFT 取决于调度 |
| 错峰混合 | TPOT P50 (ms) | 8.20 | 9.12 | 176.52 | +1835.37% | +2051.73% | 中低负载下可能降低生成延迟；无固定比例 |
| 错峰混合 | 全请求 ITL P95 (ms) | 5.81 | 55.13 | 800.61 | +1352.17% | +13679.94% | 可能改善 ITL；依赖模型、流量与硬件 |
| 错峰混合 | 首请求最大 ITL (ms) | 221.99 | 60.53 | 1115.33 | +1742.54% | +402.41% | 可能改善 ITL；无固定比例 |
| 错峰混合 | 目标模型前向调用数 | 68.00 | 64.00 | 25.00 | -60.94% | -63.24% | 一次验证最多产生 K+1 token；不含 Draft 开销 |
<!-- END GENERATED TABLE -->

变化率使用 `(第二阶段 / 对照 - 1) × 100%`。输出吞吐负值代表下降；时延负值代表改善。此处官方预期只指公开文档中的一般机制，不是针对此机器或此仓库的固定比例：阶段一对应 [vLLM Chunked Prefill](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)，阶段二对应 [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/)。

<!-- BEGIN GENERATED DETAILS -->
目标模型前向调用数按外部驱动观察 `run` 和 `run_speculative` 得出：一次普通 batch 为一次，每条推测请求的验证为一次；包含 Prefill，排除 Draft 前向。普通 batch 可同时处理多请求，而当前推测验证按请求分别执行，因而调用数不直接等于成本或单请求输出数。

- 短输入批次：第二阶段共接受 576/592 个候选（97.30%）；每轮推测验证前向中位数 25 次；三方调度步数中位数分别为 32、32、8。
- 长输入批次：第二阶段共接受 288/288 个候选（100.00%）；每轮推测验证前向中位数 12 次；三方调度步数中位数分别为 23、24、12。
- 错峰混合：第二阶段共接受 442/452 个候选（97.79%）；每轮推测验证前向中位数 19 次；三方调度步数中位数分别为 68、64、14。
<!-- END GENERATED DETAILS -->

## 官方理论与结果边界

- vLLM 官方文档指出分块 Prefill 可将 Prefill 和 Decode 同批处理，优先 Decode 可能改善 token 间延迟，并平衡计算与带宽；效果依赖预算和负载。此处不把文档当作本仓库的保证。
- vLLM 官方将推测解码列为中低请求率、带宽受限场景下可能降低 token 间延迟的方案，也明确实际收益依赖模型、流量、硬件及采样；它没有为本测试给出固定提升率。
- 推测解码论文在特定 T5-XXL 实验中报告 2–3 倍加速：[Leviathan 等，ICML 2023](https://proceedings.mlr.press/v202/leviathan23a.html)。这是该论文特定实验的实测结果，不能当作本次 Qwen3、同尺寸 Draft、RTX 5060 的预期。
- 按理想化独立接受率 α 的模型，K=4 时每次验证期望输出 token 为 `1 + α + α² + α³ + α⁴`。例如 α=0.8 时为 3.3616 token；这是简化的调用次数模型，未计 Draft 前向、重复 KV 重建、验证、串行批处理、终止和采样开销，**不是端到端加速率或官方保证**。本次实际接受率和 target forward 次数见上文，且 Draft 成本显著。
- [nano-vLLM 官方基准](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/README.md)的 RTX 4070 Laptop 实验报告 nano-vLLM 1434.13 token/s、vLLM 1361.84 token/s，约高 5.31%；这是不同引擎和硬件的实测，不是阶段二相对阶段一的理论提升。

## 复现

```bash
cd /home/ubuntu/ai-infra/minivllm/myvllm/wwvllm
mkdir -p /tmp/wwvllm-stage1-benchmark/wwvllm /tmp/wwvllm-stage2-benchmark/wwvllm
git archive e0c3da7 | tar -x -C /tmp/wwvllm-stage1-benchmark/wwvllm
git archive c511155 | tar -x -C /tmp/wwvllm-stage2-benchmark/wwvllm
find /tmp/wwvllm-stage1-benchmark/wwvllm /tmp/wwvllm-stage2-benchmark/wwvllm -type d -name __pycache__ -prune -exec rm -r {} +
PY=/home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python
SCRIPT='docs/性能对照/阶段二/scripts/three_way_compare.py'
PYTHONPATH=/home/ubuntu/ai-infra/minivllm/nano-vllm PYTHONDONTWRITEBYTECODE=1 "$PY" "$SCRIPT" --engine nano --output /tmp/nano.json
PYTHONPATH=/tmp/wwvllm-stage1-benchmark PYTHONDONTWRITEBYTECODE=1 "$PY" "$SCRIPT" --engine stage1 --output /tmp/stage1.json
PYTHONPATH=/tmp/wwvllm-stage2-benchmark PYTHONDONTWRITEBYTECODE=1 "$PY" "$SCRIPT" --engine stage2 --output /tmp/stage2.json
```

按上文顺序各运行两次，分别指定不同文件名。数据含 token 相对交付时间、目标调用统计、接收数及实际导入路径；这些计数由外部测试驱动观测，不修改任何被测引擎实现。
