"""Reproduce a small outcome-independent TACO stdio pilot subset."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parquet', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    expected = 'bee336c14dda183b1f700d54a149173418c7b3def295666159dd72c32aa8b326'
    digest = sha256(args.parquet.read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError('source must be the pinned TACO ALL/train-00000-of-00009.parquet')
    eligible = []
    for index, row in enumerate(pq.read_table(args.parquet).to_pylist()):
        try:
            tests = json.loads(row['input_output'])
        except (ValueError, TypeError):
            continue
        if (row['source'] != 'codeforces' or row['difficulty'] != 'EASY'
                or tests.get('fn_name') or not tests.get('inputs')
                or len(tests['inputs']) != len(tests.get('outputs', []))
                or 'interactive' in row['question'].lower()
                or 'constructive algorithms' in row['raw_tags']):
            continue
        eligible.append((len(row['question']), index, row))
    selected = sorted(eligible)[:16]
    if len(selected) != 16:
        raise ValueError('not enough eligible tasks')
    rows = [dict(task_id=row['url'], statement=row['question'],
                 starter_code=row['starter_code'] or '', tests=[row['input_output']],
                 source=row['source'], lineage_group=row['url'])
            for _, _, row in selected]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
    with args.output.open('x') as stream:
        stream.write(payload)
    provenance = dict(dataset='BAAI/TACO', revision='6e7429e9bcfb0e8d7aebfc719d98518f770b3985',
                      source_file='ALL/train-00000-of-00009.parquet', source_sha256=digest,
                      selection='Shortest question, then source row index; EASY Codeforces stdio; nonempty aligned full tests; exclude interactive and constructive algorithms; first 16.',
                      row_indices=[index for _, index, _ in selected],
                      task_ids=[row['task_id'] for row in rows],
                      output_sha256=sha256(payload.encode()).hexdigest(),
                      status='pilot_subset_not_formal_evidence')
    with args.output.with_suffix('.provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2)
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
