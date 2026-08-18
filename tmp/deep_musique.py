import csv,json,re,statistics
from pathlib import Path
runs={
 'closed_book': Path('outputs/musique_closed_book_qwen_api_20260520_095813_closed_book_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'sparse_rag': Path('outputs/musique_sparse_rag_qwen_api_20260520_095848_sparse_rag_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'dense_rag': Path('outputs/musique_dense_rag_qwen_api_20260520_095930_dense_rag_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'ircot_sparse': Path('outputs/musique_ircot_sparse_qwen_api_20260520_100334_ircot_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'tdca': Path('batch_outputs/musique_ans_v1.0_dev_20260520_100704'),
}
orig={}
with open('musique-main/musique-main/data/musique_ans_v1.0_dev.jsonl',encoding='utf-8') as f:
    for line in f:
        item=json.loads(line); orig[item['id']]=item

def read_summary(p):
    with open(p/'summary.csv',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    out={}
    for r in rows:
        sid=r.get('sample_id')
        r['_em']=int(float(r.get('exact_match') or 0))
        r['_soft']=int(float(r.get('soft_em') or 0)) if r.get('soft_em','')!='' else 0
        r['_f1']=float(r.get('answer_f1') or 0)
        r['_title_hit']=int(float(r.get('title_hit') or 0)) if r.get('title_hit','')!='' else 0
        out[sid]=r
    return out
summ={k:read_summary(v) for k,v in runs.items()}
ids=list(summ['tdca'].keys())
print('per-alg correct IDs count')
for alg in runs:
    print(alg, sum(summ[alg][i]['_em'] for i in ids), 'soft', sum(summ[alg][i]['_soft'] for i in ids), 'titlehit', sum(summ[alg][i]['_title_hit'] for i in ids))
print('\nby hop prefix')
for alg in runs:
    groups={}
    for i in ids:
        prefix=i.split('__',1)[0]
        groups.setdefault(prefix,[]).append(summ[alg][i]['_em'])
    print(alg, {g:f'{sum(v)}/{len(v)}={sum(v)/len(v):.2f}' for g,v in sorted(groups.items())})
print('\nby original type/level')
for alg in runs:
    groups={}
    for i in ids:
        item=orig[i]
        key=(item.get('type',''), item.get('level',''))
        groups.setdefault(key,[]).append(summ[alg][i]['_em'])
    print(alg, {str(k):f'{sum(v)}/{len(v)}={sum(v)/len(v):.2f}' for k,v in sorted(groups.items())})
print('\noverlap patterns')
for alg in ['closed_book','sparse_rag','dense_rag','ircot_sparse']:
    both=sum(summ['tdca'][i]['_em'] and summ[alg][i]['_em'] for i in ids)
    tdca_only=sum(summ['tdca'][i]['_em'] and not summ[alg][i]['_em'] for i in ids)
    alg_only=sum((not summ['tdca'][i]['_em']) and summ[alg][i]['_em'] for i in ids)
    neither=sum((not summ['tdca'][i]['_em']) and (not summ[alg][i]['_em']) for i in ids)
    print('tdca vs',alg, dict(both=both, tdca_only=tdca_only, alg_only=alg_only, neither=neither))
print('\nunique correct')
for alg in runs:
    uniq=[]
    for i in ids:
        if summ[alg][i]['_em'] and sum(summ[a][i]['_em'] for a in runs if a!=alg)==0:
            uniq.append(i)
    print(alg, len(uniq), uniq[:10])
print('\nTDCA stop reasons')
from collections import Counter
print(Counter(r.get('stop_reason','') for r in summ['tdca'].values()))
print('TDCA stop em')
for reason, cnt in Counter(r.get('stop_reason','') for r in summ['tdca'].values()).most_common():
    vals=[r['_em'] for r in summ['tdca'].values() if r.get('stop_reason','')==reason]
    print(reason, sum(vals), '/', len(vals), round(sum(vals)/len(vals),3))
print('\nTDCA title hit vs correctness')
print(Counter((r['_title_hit'],r['_em']) for r in summ['tdca'].values()))
print('\nIRCoT title hit vs correctness')
print(Counter((r['_title_hit'],r['_em']) for r in summ['ircot_sparse'].values()))
print('\nTDCA wrong but title_hit=1 examples')
for i in ids:
    r=summ['tdca'][i]
    if not r['_em'] and r['_title_hit']:
        print(i, '| q=',r['question'][:100], '| gold=',r['gold'], '| pred=',r['pred'], '| f1=',r['_f1'], '| stop=',r.get('stop_reason'))
print('\nIRCoT correct TDCA wrong examples')
for i in ids:
    if summ['ircot_sparse'][i]['_em'] and not summ['tdca'][i]['_em']:
        print(i, '| q=',summ['tdca'][i]['question'][:100], '| gold=',summ['tdca'][i]['gold'], '| tdca=',summ['tdca'][i]['pred'], '| ircot=',summ['ircot_sparse'][i]['pred'])
print('\nTDCA correct IRCoT wrong examples')
for i in ids:
    if summ['tdca'][i]['_em'] and not summ['ircot_sparse'][i]['_em']:
        print(i, '| q=',summ['tdca'][i]['question'][:100], '| gold=',summ['tdca'][i]['gold'], '| tdca=',summ['tdca'][i]['pred'], '| ircot=',summ['ircot_sparse'][i]['pred'])
