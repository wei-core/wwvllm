# 调优 02：可选的混合批次 CUDA Graph 复用

## 数据与假设

此前阶段一错峰负载中，全请求 ITL P95 约 55.70 ms；[调优 01](../01-混合批次预算/README.md) 限制混合 Prefill token 数至 128 后，错峰吞吐下降 46.4%，ITL P95 未改善，表明简单减少每步 token 数不足以解决问题。[混合执行 CPU 剖析](../02-混合执行分析/README.md)在一个 Decode=1、Prefill=511 的 step 中记录了 113 次矩阵乘、170 次 TorchDynamo 查找和 28 次变长 Attention 调用；该次 profiler CPU self 时间为 50.585 ms，但本机 CUPTI 不可用，不能获取 GPU kernel 用时。

由此提出假设：**对于重复出现的混合 batch 元数据形状，复用 CUDA Graph 可减少逐层 CPU 调用和 kernel 提交开销**。这是对已有 CUDA Graph 的应用，统一 Token Budget、调度顺序、Paged KV、Prefix Caching、Attention 算法和采样策略不变。首次捕获、图缓存显存和动态形状可能抵消收益，因此默认关闭，必须测量暖态和首轮开销。

## 方案与实现

- 新增 `enable_mixed_graph=False`、`max_mixed_graphs=16`，可由调用方显式启用并限制 LRU 缓存图的数量。
- 仅对**单 GPU、Qwen3、浮点权重、非 eager**的 Prefill + Decode 混合普通批次启用；纯 Prefill、纯 Decode、TP>1、量化、其他架构、eager 模式及推测验证保持原执行路径。
- 以本轮 token 数、请求数、`max_seqlen_q/k` 和 block table 宽度为键；每次拷贝 input IDs、positions、变长长度、slot mapping 和 block table 到图专属输入缓冲区。捕获模型 forward，回放后在当前 Context 下按本轮 `sample_indices` 计算 logits。
- 未命中时用真实输入预运行并捕获一次，再执行回放；首次开销和 GPU 计算增加。缓存达到上限后淘汰最久未使用的图；不同实际 batch 数据相同形状仍更新元数据。缓存数量受限不代表 CUDA allocator 保留显存一定按数量线性释放。
- 图共享现有 CUDA Graph 内存池；引擎采用单个工作线程、一次一个 batch，并在混合批次采样后同步取回 token，不将前一批 graph 输出提供给下一批使用。共享池的并发和输出生命周期限制参照 [PyTorch CUDA Graph 内存管理文档](https://docs.pytorch.org/docs/stable/notes/cuda.html#graph-memory-management)。实验尚未验证多线程并发、TP=2 或非 CUDA Graph 兼容的其他环境，因此这些路径不启用此优化。

使用方式：

```python
from wwvllm import LLM
llm = LLM('/path/to/Qwen3', enable_mixed_graph=True, max_mixed_graphs=16)
```

保持默认关闭是有意选择：冷启动和高动态形状场景可能更慢。捕获过程中如设备、依赖或显存导致错误，该异常会传播；调用方可关闭本开关使用原 eager 混合路径。实现变更见 [implementation.patch.gz](implementation.patch.gz) 和仓库 `config.py`、`engine/model_runner.py`。

## 基准方案

基线固定 `ae4dff1`；对照是在同一代码基础上启用此开关。对称顺序：基线 → 图复用 → 图复用 → 基线；每个进程对各负载先执行一轮首遇形状的预热，再测 3 轮，即每种负载、每组 6 次暖态。首轮预热作为冷态辅助数据保存，不混入暖态表。

硬件与参数：RTX 5060 8 GiB、Qwen3-0.6B BF16、TP=1、PyTorch 2.9.1+cu128、FlashAttention 2.8.3；GPU 与桌面共享，未锁定频率。`max_num_batched_tokens=512`、`prefill_chunk_size=512`、`max_model_len=2048`、`max_num_seqs=8`、`gpu_memory_utilization=0.60`、temperature 0.6、ignore EOS；推测解码和量化关闭。

- 短输入：4 个请求，每个 128 输入/32 输出。
- 长输入：4 个请求，每个 1024 输入/16 输出。
- 错峰：先提交 64 输入/64 输出请求，其输出达到 4 token 后，再提交 2 个 1024 输入/16 输出请求。

统一驱动在 `step()` 完成时记录交付时间；吞吐为输出 token/首请求到全部完成的墙钟时间。TTFT/TPOT P50、ITL P95 为每轮样本统计后再取 6 轮中位数。不是 HTTP 压测，3–4 请求不足以表征生产尾延迟；不同模式与种子的输出按哈希留档。

## 暖态实测结果

| 场景 | 指标 | 基线 | 图复用 | 变化 `(图/基线-1)` |
| --- | --- | ---: | ---: | ---: |
| 短输入 | 输出吞吐 token/s | 601.45 | 573.04 | -4.72% |
| 短输入 | TTFT P50 ms | 56.88 | 58.72 | +3.24% |
| 短输入 | TPOT P50 ms | 5.05 | 5.23 | +3.57% |
| 短输入 | ITL P95 ms | 5.44 | 5.84 | +7.29% |
| 长输入 | 输出吞吐 token/s | 108.88 | 179.56 | **+64.93%** |
| 长输入 | TTFT P50 ms | 334.03 | 212.14 | **-36.49%** |
| 长输入 | TPOT P50 ms | 15.87 | 8.54 | **-46.20%** |
| 长输入 | ITL P95 ms | 55.83 | 25.47 | **-54.38%** |
| 错峰 | 输出吞吐 token/s | 153.85 | 215.55 | **+40.10%** |
| 错峰 | TTFT P50 ms | 170.18 | 67.96 | **-60.07%** |
| 错峰 | TPOT P50 ms | 9.02 | 6.19 | **-31.39%** |
| 错峰 | ITL P95 ms | 54.59 | 12.62 | **-76.89%** |

变化率负值对延迟表示缩短，对吞吐表示下降。短输入没有混合步，图路径不会执行；其约 5% 的变化不能合理归因于图复用，可能来自运行波动。长输入每轮 7 次、错峰每轮 5 次混合步在暖态全部命中现有图；两组对应的 18 轮随机输出哈希一致，只表明该样本在本环境匹配，不构成所有随机模型的分布证明。

## 冷态与显存代价

| 首轮负载（每组 2 次进程的中位数） | 基线墙钟时间 | 图复用墙钟时间 | 图捕获次数 |
| --- | ---: | ---: | ---: |
| 长输入批次 | 0.813 s | 1.698 s | 7 |
| 错峰混合 | 0.626 s | 0.993 s | 5 |

这些首轮是模型初始化后、同进程短输入预热之后首次遇到对应混合形状，不包含完整模型加载时间，也可能包含其他首次形状编译；不能当作线上首请求的通用指标。图复用的两次进程最终各缓存 12 个图，之后测量轮次没有新增捕获或淘汰。基线峰值 `torch.cuda.max_memory_allocated()` 为 3,829,577,728 字节，图复用为 3,840,228,352 字节（+10,650,624 字节，约 +10.16 MiB）；reserved 峰值分别为 3,909,091,328 和 3,923,771,392 字节。这些数值依赖图形状、自动 KV 分配与 GPU 共享状态，不保证其他负载的显存上界。

## 正确性与边界

- GPU 测试比较小 chunk、跨 256-token 块、参考路径与图路径的贪心结果，以及一个真实混合批次采样 logits 的数值；将缓存限制设为 2 并验证图淘汰仍输出正确。
- 其他测试覆盖调度、异步、推测采样、与 Speculative 请求共存、原有 Qwen3 GPU、量化算子、Llama/Qwen3-MoE 小模型、合成 AWQ/GPTQ 和真实 MoE（按本机条件运行）。
- 本机全量命令设置 `WWVLLM_TEST_MODEL`、`WWVLLM_STAGE3_GPU=1`、`WWVLLM_MOE_MODEL` 后得到 **39 通过、2 失败**；失败分别为 INT8 Target + Draft 和真实 MoE 初始化时 KV 缓存显存不足。两项随后在 GPU 占用下降时分别重跑，**各 1 项通过**。不能将其表述成一次完整套件全部通过；Graph 输出/logits/淘汰及 Graph + Speculative 共存用例在全量运行中通过。
- 输出匹配仅针对测试请求；尚未覆盖真实多 GPU、生产异步压测、异常捕获后恢复、不同 CUDA/FlashAttention 版本、长时间图淘汰压力和动态批次分布。当前开关默认关闭，对未覆盖的多 GPU、量化、其他模型和 eager 路径不会启用图复用。

## 复现

原始数据在 [data/](data/)，脚本为 [compare.py](scripts/compare.py)。用 `git archive ae4dff1` 导出基线至独立临时目录；对另一个副本以 `gzip -dc implementation.patch.gz | patch -p1` 应用 [implementation.patch.gz](implementation.patch.gz)，分别指定各自的 `PYTHONPATH`。基线使用 `--engine stage1`，实验使用 `--engine stage1 --mixed-graph`；每组各执行两次并使用不同输出文件，按上文对称顺序运行。保留的 JSON 包含逐 token 时间、图捕获/回放计数、预热轮及显存峰值；脚本记录的是过程实际观测，不对引擎进行参数搜索。
