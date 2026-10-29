"""Render the comparison table and observations from the six result files."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
CASES = [('short_batch', '短输入批次'), ('long_batch', '长输入批次'), ('staggered', '错峰混合')]
METRICS = [
    ('output_tokens_per_s', '输出吞吐 (token/s)', 1, 'Draft 足够快时可能提高吞吐；无固定比例'),
    ('ttft_p50_s', 'TTFT P50 (ms)', 1000, '推测主要作用于 Decode；TTFT 取决于调度'),
    ('tpot_p50_s', 'TPOT P50 (ms)', 1000, '中低负载下可能降低生成延迟；无固定比例'),
    ('itl_p95_s', '全请求 ITL P95 (ms)', 1000, '可能改善 ITL；依赖模型、流量与硬件'),
    ('lead_itl_max_s', '首请求最大 ITL (ms)', 1000, '可能改善 ITL；无固定比例'),
    ('target_forwards', '目标模型前向调用数', 1, '一次验证最多产生 K+1 token；不含 Draft 开销'),
]


def main():
    all_rows = {}
    for engine in ('nano', 'stage1', 'stage2'):
        files = sorted(DATA.glob(f'three-{engine}-*.json'))
        if len(files) != 2:
            raise ValueError(f'{engine}: expected two result files')
        runs = [json.loads(p.read_text()) for p in files]
        if any(run['engine'] != engine or len(run['results']) != 9 for run in runs):
            raise ValueError(f'{engine}: invalid source or result count')
        for case, _ in CASES:
            rows = [row for run in runs for row in run['results'] if row['case'] == case]
            if len(rows) != 6:
                raise ValueError(f'{engine}/{case}: expected six runs')
            all_rows[(engine, case)] = rows
    table = ['| 场景 | 指标 | 原版 nano-vLLM | 第一阶段 | 第二阶段 | 二阶段对一阶段 | 二阶段对原版 | 官方理论预期 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    details = []
    for case, label in CASES:
        for key, metric, factor, official in METRICS:
            values = [statistics.median(r[key] for r in all_rows[(engine, case)]) * factor
                      for engine in ('nano', 'stage1', 'stage2')]
            fmt = lambda v: f'{v:.2f}'
            nano, stage1, stage2 = values
            change1 = (stage2 / stage1 - 1) * 100
            change0 = (stage2 / nano - 1) * 100
            table.append(f'| {label} | {metric} | {fmt(nano)} | {fmt(stage1)} | {fmt(stage2)} | {change1:+.2f}% | {change0:+.2f}% | {official} |')
        rows = all_rows[('stage2', case)]
        accepted, proposed = sum(r['accepted'] for r in rows), sum(r['proposed'] for r in rows)
        spec = statistics.median(r['verification_forwards'] for r in rows)
        counts = [statistics.median(r['steps'] for r in all_rows[(e, case)]) for e in ('nano', 'stage1', 'stage2')]
        details.append(f'- {label}：第二阶段共接受 {accepted}/{proposed} 个候选（{accepted / proposed:.2%}）；每轮推测验证前向中位数 {spec:g} 次；三方调度步数中位数分别为 {counts[0]:g}、{counts[1]:g}、{counts[2]:g}。')
    report = (ROOT / 'README.md').read_text()
    def replace_region(text, name, content):
        begin = f'<!-- BEGIN GENERATED {name} -->'
        end = f'<!-- END GENERATED {name} -->'
        if text.count(begin) != 1 or text.count(end) != 1:
            raise ValueError(f'missing or repeated {name} markers')
        before, remaining = text.split(begin, 1)
        _, after = remaining.split(end, 1)
        return before + begin + '\n' + content + '\n' + end + after
    explanation = ('目标模型前向调用数按外部驱动观察 `run` 和 `run_speculative` 得出：一次普通 batch 为一次，'
                   '每条推测请求的验证为一次；包含 Prefill，排除 Draft 前向。普通 batch 可同时处理多请求，'
                   '而当前推测验证按请求分别执行，因而调用数不直接等于成本或单请求输出数。\n\n' + '\n'.join(details))
    report = replace_region(report, 'TABLE', '\n'.join(table))
    report = replace_region(report, 'DETAILS', explanation)
    (ROOT / 'README.md').write_text(report)
    print('\n'.join(table))
    print(explanation)


if __name__ == '__main__':
    main()
