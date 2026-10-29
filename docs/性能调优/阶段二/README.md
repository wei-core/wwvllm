# 第二阶段性能调优

- [01 Draft KV 复用实验](01-Draft缓存/README.md)：依据 Draft 反复计算提出缓存方案，记录正确性检查、收益和退化；当前未合入服务。

- [02 同系列异规模 Draft 重测](02-异规模Draft/README.md)：Qwen3-1.7B / 0.6B 的普通解码、重建及缓存对照，包含 greedy 分歧与目标 logits 诊断。
