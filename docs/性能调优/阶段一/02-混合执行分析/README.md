# 混合执行剖析：下一轮调优依据

## 动机和方法

调优 01 将混合 Prefill token 上限从 512 降至 128 后，混合次数增加、吞吐下降，但单步 ITL P95 约 54–58 ms 并未随 token 数显著缩短。为避免继续仅凭假设改变参数，使用 `torch.profiler` 对基线 `ae4dff1` 上一次预热后的真实混合 `step()` 作诊断：[脚本](scripts/profile_mixed.py)、[记录输出](data/profile.txt)（已移除行尾空格）。这不是性能基准；profiler 自身的记录开销、同步和统计方式会影响测量。

当次 step 为 1 个 Decode token 和 511 个 Prefill token，共 512 个 token。CPU profiler 的 self CPU 总计 50.585 ms；其中记录 113 次 `aten::mm` 调用、170 次 TorchDynamo cache lookup 和 28 次 FlashAttention varlen 调用。各种编译区域和 wrapper 的 CPU 调度在表中占有可见比例；这些值不能直接相加成独立的 GPU 用时，也不能证明因果。

**限制**：当前环境 CUPTI 初始化返回 `CUPTI_ERROR_INVALID_DEVICE`，所以 CUDA profiler 活动缺失，原输出中所谓按 CUDA 排序的表不含有效 GPU 耗时。上述诊断只能说明当前观测到较多 CPU 调用和提交活动，不能据此判断 GPU kernel 时间或 GPU 利用率瓶颈。

## 下一轮思路

基线的纯 Decode 已使用 CUDA Graph，但含 Prefill 的混合 forward 总走 eager，并反复经过每层的 Python/编译包装和内核提交。可探索在不改变统一调度、Paged KV、混合 Attention 数学或采样语义的前提下，**针对可重复的混合批次形状复用 CUDA Graph**。必须保留 eager 回退，固定元数据缓冲区并确保 slot、block table、序列长度和采样索引每轮正确更新；同时检验缓存写入、前缀复用、动态形状、TP 和推测验证的安全性。

该思路尚未得到性能或正确性验证。首次 capture、图缓存占用、固定上界导致的额外计算和动态形状过多都可能抵消收益；如果实测不改善或存在状态风险，应拒绝而不是宣称优化成功。
