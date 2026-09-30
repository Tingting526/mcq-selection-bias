"""Download the pinned official PriDe source datasets, with provenance and audit.

Keeps the original CSVs intact. Normalized JSONL files are reference datasets,
NOT four-option preprocessing artifacts; CSQA retains all five answers.
"""
from __future__ import annotations
import ast
import csv
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request
from _bootstrap import PROJECT_ROOT

COMMIT = '2ae2f40c77006f7a00a4675bdfb30151c2406691'
URL = f'https://codeload.github.com/chujiezheng/LLM-MCQ-Bias/tar.gz/{COMMIT}'
ROOT = PROJECT_ROOT / 'data/reference/zheng2024'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def unchanged_or_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'Refusing to overwrite changed reference: {path}')
    else:
        path.write_bytes(content)


def filter_constants(code):
    # Read only literal lists from upstream. Never execute downloaded scripts.
    result = {}
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {'BAD_OPTIONS', 'REFER_OPTIONS'}:
                    result[target.id] = ast.literal_eval(node.value)
    return result['BAD_OPTIONS'], result['REFER_OPTIONS']


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'source.tar.gz'
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=120) as response:
            unchanged_or_write(archive, response.read())
    with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as bundle:
        for member in bundle.getmembers():
            parts = Path(member.name).parts[1:]
            if member.isfile() and parts and '..' not in parts:
                unchanged_or_write(ROOT / 'upstream' / Path(*parts), bundle.extractfile(member).read())
    upstream = ROOT / 'upstream/code'
    bad, refer = filter_constants((upstream / 'utils.py').read_text())
    datasets = []
    for task, count in [('mmlu', 4), ('arc', 4), ('csqa', 5)]:
        for split in ['dev', 'test']:
            records, excluded = [], []
            files = []
            for path in sorted((upstream / f'data_{task}' / split).glob('*.csv')):
                files.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path.read_bytes())))
                subject = path.stem.removesuffix('_' + split)
                with path.open(newline='', encoding='utf-8') as stream:
                    for index, row in enumerate(csv.reader(stream)):
                        if len(row) != count + 2 or row[-1] not in 'ABCDE'[:count]:
                            raise ValueError(f'Invalid row: {path}:{index}')
                        record = dict(sample_id=f'zheng2024:{task}:{split}:{subject}:{index}',
                                      dataset=task, split=split, subject=subject, question=row[0],
                                      options=row[1:-1], gold_index='ABCDE'.index(row[-1]))
                        record['excluded_by_upstream_filter'] = (
                            any(x in option for x in bad for option in record['options']) or
                            any(x in option.lower() for x in refer for option in record['options']))
                        records.append(record)
                        if record['excluded_by_upstream_filter']:
                            excluded.append(record['sample_id'])
            target = ROOT / 'normalized' / f'{task}_{split}.jsonl'
            content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records).encode()
            unchanged_or_write(target, content)
            datasets.append(dict(dataset=task, repository_split=split, options=count,
                                 original_rows=len(records), upstream_filter_excluded=len(excluded),
                                 retained_rows=len(records)-len(excluded),
                                 normalized_path=str(target.relative_to(ROOT)), sha256=sha(content),
                                 source_files=files, excluded_ids=excluded))
    manifest = dict(source='https://github.com/chujiezheng/LLM-MCQ-Bias', commit=COMMIT,
                    archive_url=URL, archive_sha256=sha(archive.read_bytes()),
                    normalization='Preserve original text, order and labels; annotate original utils.py filter, do not remove rows.',
                    csqa_split_note='Repository test = shuffled official dev/validation minus 5 few-shot examples (seed 23); not hidden test.',
                    datasets=datasets)
    unchanged_or_write(ROOT / 'manifest.json', (json.dumps(manifest, indent=2)+'\n').encode())
    for data in datasets:
        print(f"{data['dataset']} {data['repository_split']}: {data['original_rows']} rows, "
              f"{data['options']} options, {data['retained_rows']} after upstream filter")
    print(f'Manifest: {ROOT / "manifest.json"}')


if __name__ == '__main__':
    main()
