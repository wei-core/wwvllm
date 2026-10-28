# 阶段一性能对照：原版 nano-vLLM 与阶段一 wwvllm

测试时间：2026-10-28（北京时间）。这是一组固定配置的实际测试，不包含性能调优。原始数据文件均在本目录：每种引擎各 2 次进程启动，每次对每个负载预热一次后测量 3 次；顺序为 nano → wwvllm → wwvllm → nano。合计每种负载、每个引擎 6 次测量。`*_run2.json` 额外保存逐 token 相对提交时间和最长 token 间隔；`*_run1.json` 保留较早的观测器字段。

## 测试对象和环境

- 原版：本机 [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/tree/bb823b3e06983d71485a8e1f23715ebd87d98ef8)，commit `bb823b3e06983d71485a8e1f23715ebd87d98ef8`。其 `nanovllm/` 内所有 Python 文件与本仓库改造前的 `57ac1cd` 对应 Python 文件逐字节相同；原仓库 README、bench.py 和 example.py 有本地修改，未作为被测源码或测试脚本。
- 第一阶段：从本仓库 `e0c3da7` 独立导出，不使用其后第二、三阶段代码。原始数据 `package_path` 显示实际导入位置。
- GPU：NVIDIA GeForce RTX 5060 8 GiB，TP=1。启动前 GPU 有约 2 GiB 显存占用（桌面 Xwayland）；测试未对其他进程隔离，也未锁定 GPU 时钟，测量期间负载和频率可能变化。
- 模型：本机 Qwen3-0.6B BF16，`model.safetensors` SHA-256 为 `f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`。
- 软件：PyTorch 2.9.1+cu128、Transformers 4.57.3、FlashAttention 2.8.3、CUDA Graph 开启（`enforce_eager=False`）。同一 conda 环境、同一 GPU 和同一模型目录。
- 共同配置：`max_num_batched_tokens=512`、`max_num_seqs=8`、`max_model_len=2048`、`gpu_memory_utilization=0.60`、`temperature=0.6`、`ignore_eos=True`。阶段一 `prefill_chunk_size=512` 使用该阶段默认值。512 是预先固定、用以覆盖超过预算的 Prefill 的共同测试参数，不是搜索得到的最优参数；不代表默认配置或所有负载。
- 请求使用相同种子的随机 token ID；各轮更换 prompt，防止上一轮 Prefix Cache 命中。每个请求固定输出长度，所有请求均完成。随机采样下两引擎调度顺序及随机数消费不同，输出内容不要求一致，本测试不是生成等价性或质量评估。

## 负载和计时

- **短输入批次**：4 个请求同时提交，每个 128 输入、32 输出 token。
- **长输入批次**：4 个请求同时提交，每个 1024 输入、16 输出 token。
- **错峰混合**：先提交一个 64 输入、64 输出请求；它产生 4 个 token 后，提交两个各为 1024 输入、16 输出的请求。这是以生成进度触发的可复现错峰，不是按固定请求到达率的线上服务压测。

使用相同的外部 Python 驱动 `add_request()`/`step()`，在每一步返回时观察每个请求已生成 token 数并使用 `time.perf_counter()` 打点；两侧均包含相同观测开销，起点在首次请求提交前，终点在全部请求完成且 CUDA 同步之后。随机输入构造和模型初始化、编译预热不计入。由于原版无对等异步服务接口，测试没有比较异步 API、网络、序列化或真实客户端排队开销。

- 吞吐：总输出 token / 从首请求提交到全部完成的墙钟时间。
- TTFT P50：单轮各请求从提交到首 token 的延迟中位数；表中再对 6 轮中位数取中位数。
- TPOT P50：单请求 `(末 token 时间 - 首 token 时间) / (输出数 - 1)`，同样先对请求再对 6 轮取中位数。
- ITL P95：每轮收集所有请求的相邻 token 间隔，插值求 P95；表中为 6 轮结果的中位数。一个极长停顿可能少于全部间隔的 5%，因此可能不被 P95 反映。
- 首请求最大 ITL：每轮第一条请求相邻 token 间隔的最大值，仅后 3 轮记录，表中为这 3 轮的中位数。
- 变化率统一为 `(阶段一 / 原版 - 1) × 100%`；吞吐负值代表下降，延迟负值代表缩短。微小差异不能视为统计显著。单轮只有 3–4 请求，样本不足以推断生产环境尾延迟。

## 对比结果

| 场景 | 指标 | 原版 nano-vLLM | 阶段一 wwvllm | 实测变化 | 官方理论预期 |
| --- | --- | ---: | ---: | ---: | --- |
| 短输入批次 | 输出吞吐 (token/s) | 618.93 | 611.87 | -1.14% | 混合计算可能提高吞吐；无固定提升率 |
| 短输入批次 | TTFT P50 (ms) | 55.70 | 56.46 | +1.37% | 受 Prefill 预算与调度影响；无固定提升率 |
| 短输入批次 | TPOT P50 (ms) | 4.87 | 4.93 | +1.07% | Decode 优先可能改善生成延迟；无固定提升率 |
| 短输入批次 | 全请求 ITL P95 (ms) | 5.25 | 5.49 | +4.40% | Decode 优先可能改善 token 间延迟；无固定提升率 |
| 短输入批次 | 首请求最大 ITL (ms，3 轮) | 5.46 | 5.81 | +6.47% | Decode 优先可能改善 token 间延迟；无固定提升率 |
| 长输入批次 | 输出吞吐 (token/s) | 118.46 | 108.32 | -8.56% | 混合计算可能提高吞吐；无固定提升率 |
| 长输入批次 | TTFT P50 (ms) | 283.17 | 334.25 | +18.04% | 受 Prefill 预算与调度影响；无固定提升率 |
| 长输入批次 | TPOT P50 (ms) | 17.08 | 16.06 | -5.98% | Decode 优先可能改善生成延迟；无固定提升率 |
| 长输入批次 | 全请求 ITL P95 (ms) | 11.83 | 57.67 | +387.65% | Decode 优先可能改善 token 间延迟；无固定提升率 |
| 长输入批次 | 首请求最大 ITL (ms，3 轮) | 359.05 | 61.55 | -82.86% | Decode 优先可能改善 token 间延迟；无固定提升率 |
| 错峰混合 | 输出吞吐 (token/s) | 163.24 | 151.27 | -7.33% | 混合计算可能提高吞吐；无固定提升率 |
| 错峰混合 | TTFT P50 (ms) | 112.47 | 169.96 | +51.11% | 受 Prefill 预算与调度影响；无固定提升率 |
| 错峰混合 | TPOT P50 (ms) | 8.40 | 9.15 | +9.00% | Decode 优先可能改善生成延迟；无固定提升率 |
| 错峰混合 | 全请求 ITL P95 (ms) | 5.36 | 55.70 | +939.67% | Decode 优先可能改善 token 间延迟；无固定提升率 |
| 错峰混合 | 首请求最大 ITL (ms，3 轮) | 233.57 | 62.01 | -73.45% | Decode 优先可能改善 token 间延迟；无固定提升率 |

表中官方理论预期依据 [vLLM 的 Chunked Prefill 文档](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)，描述的是通用设计预期；官方没有给出适用于本项目、硬件和负载的理论提升百分比，因此不能填入一个可比数值。

长输入批次中，阶段一每轮实际出现 7 个 Prefill/Decode 混合调度步；错峰混合中每轮出现 5 个，原版都为 0。原版的长 Prefill 导致少数很长的停顿，阶段一将其分散为多次较短的混合步骤：这组结果中首请求最大停顿缩短，但较多中等长度间隔使全请求 ITL P95 变差。以上是对此工作负载的观察和可能机制，不证明所有在线请求都会得到相同改善。单次运行随 GPU 状态有波动，见 JSON 原始记录。

## 官方理论提示和可比范围

[vLLM 官方优化文档的 Chunked Prefill 说明](https://docs.vllm.ai/en/latest/configuration/optimization/#chunked-prefill)指出，Decode 优先及 Prefill/Decode 同批有机会改善 token 间延迟并平衡计算和带宽利用；效果受预算、负载和硬件影响。该说明是 vLLM 的一般设计预期，不是本项目阶段一的官方实测结果或固定提升比例。本次不进行参数调优，也不能将其他系统的推荐参数当成本机性能结论。

[nano-vLLM 官方固定版本 README](https://github.com/GeeeekExplorer/nano-vllm/blob/bb823b3e06983d71485a8e1f23715ebd87d98ef8/README.md)报告 RTX 4070 Laptop、Qwen3-0.6B、256 条随机长度序列下 Nano-vLLM 为 1434.13 token/s、vLLM 为 1361.84 token/s，前者约高 5.31%。这是一组**官方实测的 nano-vLLM 对 vLLM**，不是阶段一改造的理论提升，也不是本次 RTX 5060 测试；硬件、批次、长度和引擎均不同，不能直接比较。

## 复现

```bash
cd /home/ubuntu/ai-infra/minivllm/myvllm/wwvllm
mkdir -p /tmp/wwvllm-stage1-benchmark/wwvllm
git archive e0c3da7 | tar -x -C /tmp/wwvllm-stage1-benchmark/wwvllm
find /tmp/wwvllm-stage1-benchmark/wwvllm -type d -name __pycache__ -prune -exec rm -r {} +
PYTHONPATH=/home/ubuntu/ai-infra/minivllm/nano-vllm PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python docs/性能对照/阶段一/scripts/stage1_compare.py --engine nano --output /tmp/nano.json
PYTHONPATH=/tmp/wwvllm-stage1-benchmark PYTHONDONTWRITEBYTECODE=1 \
  /home/ubuntu/miniconda3/envs/ai-nanovllm/bin/python docs/性能对照/阶段一/scripts/stage1_compare.py --engine stage1 --output /tmp/stage1.json
```

要复现交替顺序，每个引擎分别运行两次；每次输出文件应使用不同名称。不同 GPU 时间片、可用显存和温度会影响结果。首次 nano 测量后原引擎 `atexit` 钩子因显式释放而打印一次重复释放异常；数据在退出前已完整写入，之后的观测脚本解除该钩子。它不影响计时，但该两轮使用的观测器在逐 token 数据字段上有差异，如文首所述。
