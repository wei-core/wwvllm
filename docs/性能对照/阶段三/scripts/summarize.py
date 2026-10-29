"""Render a four-way comparison from six recorded runs per engine."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
ENGINES = ('nano', 'stage1', 'stage2', 'stage3')
CASES = [('short_batch', '短输入批次'), ('long_batch', '长输入批次'), ('staggered', '错峰混合')]
METRICS = [
    ('output_tokens_per_s', '输出吞吐 (token/s)', 1, 'Draft 足够快时推测解码可能提高吞吐；无固定提升率'),
    ('ttft_p50_s', 'TTFT P50 (ms)', 1000, '推测主要作用于 Decode；受 Prefill 调度影响'),
    ('tpot_p50_s', 'TPOT P50 (ms)', 1000, '中低负载下推测解码可能降低生成延迟'),
    ('itl_p95_s', '全请求 ITL P95 (ms)', 1000, '可能改善 ITL；依赖模型、流量与硬件'),
    ('lead_itl_max_s', '首请求最大 ITL (ms)', 1000, '可能改善 ITL；无固定比例'),
    ('target_forwards', '目标模型前向调用数', 1, 'K=4 时一次验证最多交付 5 token，不含 Draft 开销'),
]


def replace_region(text, name, content):
    begin, end = f'<!-- BEGIN {name} -->', f'<!-- END {name} -->'
    if text.count(begin) != 1 or text.count(end) != 1:
        raise ValueError(f'missing or repeated {name} markers')
    before, remaining = text.split(begin, 1)
    _, after = remaining.split(end, 1)
    return before + begin + '\n' + content + '\n' + end + after


def main():
    rows = {}
    for engine in ENGINES:
        files = sorted(DATA.glob(f'four-{engine}-*.json'))
        if len(files) != 2:
            raise ValueError(f'{engine}: expected two result files')
        runs = [json.loads(p.read_text()) for p in files]
        if any(run['engine'] != engine or len(run['results']) != 9 for run in runs):
            raise ValueError(f'{engine}: invalid source or result count')
        for case, _ in CASES:
            selected = [row for run in runs for row in run['results'] if row['case'] == case]
            if len(selected) != 6:
                raise ValueError(f'{engine}/{case}: expected six rounds')
            rows[(engine, case)] = selected
    table = ['| 场景 | 指标 | 原版 | 第一阶段 | 第二阶段 | 第三阶段 | 三对原版 | 三对一 | 三对二 | 官方理论预期 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    details = []
    for case, label in CASES:
        for key, metric, factor, official in METRICS:
            values = [statistics.median(r[key] for r in rows[(engine, case)]) * factor for engine in ENGINES]
            values_text = ' | '.join(f'{v:.2f}' for v in values)
            delta_text = ' | '.join(f'{(values[3] / v - 1) * 100:+.2f}%' for v in values[:3])
            table.append(f'| {label} | {metric} | {values_text} | {delta_text} | {official} |')
        counts = [statistics.median(r['steps'] for r in rows[(engine, case)]) for engine in ENGINES]
        details.append(f'- {label}：四组每轮调度步数中位数依次为 {counts[0]:g}、{counts[1]:g}、{counts[2]:g}、{counts[3]:g}。')
        for engine, label_engine in [('stage2', '第二阶段'), ('stage3', '第三阶段')]:
            selected = rows[(engine, case)]
            accepted, proposed = sum(x['accepted'] for x in selected), sum(x['proposed'] for x in selected)
            verifies = statistics.median(x['verification_forwards'] for x in selected)
            if proposed <= 0:
                raise ValueError(f'no speculative proposals for {engine}/{case}')
            details.append(f'  {label_engine}：候选接受 {accepted}/{proposed}（{accepted/proposed:.2%}），每轮验证前向中位数 {verifies:g}。')
    report = (ROOT / 'README.md').read_text()
    report = replace_region(report, 'FOUR WAY TABLE', '\n'.join(table))
    report = replace_region(report, 'FOUR WAY DETAILS', '\n'.join(details))
    (ROOT / 'README.md').write_text(report)
    print('\n'.join(table))
    print('\n'.join(details))


if __name__ == '__main__':
    main()
