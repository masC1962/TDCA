import csv, json, re
from pathlib import Path
root=Path('.')
paths=[]
for p in Path('outputs').glob('musique_*'):
    if (p/'aggregate.json').exists() and (p/'summary.csv').exists():
        alg='unknown'
        name=p.name
        if 'closed_book' in name: alg='closed_book'
        elif 'sparse_rag' in name: alg='sparse_rag'
        elif 'dense_rag' in name: alg='dense_rag'
        elif 'ircot' in name: alg='ircot_sparse'
        paths.append((alg,p.stat().st_mtime,p))
if Path('batch_outputs/musique_ans_v1.0_dev_20260520_100704/aggregate.json').exists():
    p=Path('batch_outputs/musique_ans_v1.0_dev_20260520_100704')
    paths.append(('tdca',p.stat().st_mtime,p))
# latest per algorithm
latest={}
for alg,mt,p in paths:
    if alg not in latest or mt>latest[alg][0]: latest[alg]=(mt,p)
print('LATEST')
for alg in ['closed_book','sparse_rag','dense_rag','ircot_sparse','tdca']:
    if alg in latest:
        p=latest[alg][1]
        agg=json.loads((p/'aggregate.json').read_text(encoding='utf-8'))
        print(alg, p, agg)
print('\nALL musique runs')
for alg,mt,p in sorted(paths, key=lambda x:(x[0],x[1])):
    agg=json.loads((p/'aggregate.json').read_text(encoding='utf-8'))
    print(alg, p.name, 'count',agg.get('count'),'em',agg.get('exact_match'),'f1',agg.get('answer_f1'),'title_hit',agg.get('title_hit'))
