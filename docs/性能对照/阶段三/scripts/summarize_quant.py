"""Render stage-three isolated quantization measurements."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'


def main():
    modes = ('bf16', 'int8', 'int4')
    results = [json.loads((DATA / f'feature-{mode}.json').read_text()) for mode in modes]
    for mode, data in zip(modes, results):
        if data['feature_mode'] != mode or len(data['results']) != 3:
            raise ValueError(f'invalid results for {mode}')
        if any(row['case'] != 'short_batch' or row['output_tokens'] != 128 for row in data['results']):
            raise ValueError(f'invalid workload for {mode}')
    speed0 = statistics.median(row['output_tokens_per_s'] for row in results[0]['results'])
    bytes0 = results[0]['model_tensor_bytes']
    table = ['| 第三阶段模式 | 输出吞吐 (token/s) | 相对 BF16 | TTFT P50 (ms) | TPOT P50 (ms) | 模型张量字节 | 相对 BF16 | 位宽理论提示 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    theory = ('16-bit 基准', '仅权重本体约 50%，不含元数据', '仅权重本体约 25%，不含元数据')
    for mode, data, expected in zip(modes, results, theory):
        speed = statistics.median(row['output_tokens_per_s'] for row in data['results'])
        ttft = statistics.median(row['ttft_p50_s'] for row in data['results']) * 1000
        tpot = statistics.median(row['tpot_p50_s'] for row in data['results']) * 1000
        num_bytes = data['model_tensor_bytes']
        table.append(f'| {mode.upper()} | {speed:.2f} | {(speed / speed0 - 1) * 100:+.2f}% | {ttft:.2f} | {tpot:.2f} | {num_bytes:,} | {(num_bytes / bytes0 - 1) * 100:+.2f}% | {expected} |')
    report = (ROOT / 'README.md').read_text()
    begin, end = '<!-- BEGIN QUANT TABLE -->', '<!-- END QUANT TABLE -->'
    if report.count(begin) != 1 or report.count(end) != 1:
        raise ValueError('invalid quantization table markers')
    before, remaining = report.split(begin, 1)
    _, after = remaining.split(end, 1)
    (ROOT / 'README.md').write_text(before + begin + '\n' + '\n'.join(table) + '\n' + end + after)
    print('\n'.join(table))


if __name__ == '__main__':
    main()
