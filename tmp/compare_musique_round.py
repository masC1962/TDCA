import csv,json,statistics
from pathlib import Path
old={
 'closed_book': Path('outputs/musique_closed_book_qwen_api_20260520_095813_closed_book_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'sparse_rag': Path('outputs/musique_sparse_rag_qwen_api_20260520_095848_sparse_rag_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'dense_rag': Path('outputs/musique_dense_rag_qwen_api_20260520_095930_dense_rag_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'ircot_sparse': Path('outputs/musique_ircot_sparse_qwen_api_20260520_100334_ircot_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'tdca': Path('batch_outputs/musique_ans_v1.0_dev_20260520_100704'),
}
new={
 'closed_book': Path('outputs/musique_closed_book_qwen_api_20260520_104234_closed_book_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'sparse_rag': Path('outputs/musique_sparse_rag_qwen_api_20260520_104311_sparse_rag_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'dense_rag': Path('outputs/musique_dense_rag_qwen_api_20260520_104345_dense_rag_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'ircot_sparse': Path('outputs/musique_ircot_sparse_qwen_api_20260520_104421_ircot_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'tdca': Path('batch_outputs/musique_ans_v1.0_dev_20260520_104731'),
}
def agg(p): return json.loads((p/'aggregate.json').read_text(encoding='utf-8'))
def rows(p):
    with open(p/'summary.csv',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
def rowmap(p):
    m={}
    for r in rows(p):
        r['_em']=int(float(r.get('exact_match') or 0)); r['_soft']=int(float(r.get('soft_em') or 0)); r['_f1']=float(r.get('answer_f1') or 0); r['_title_hit']=int(float(r.get('title_hit') or 0));
        m[r['sample_id']]=r
    return m
print('METRICS old -> new')
for alg in ['closed_book','sparse_rag','dense_rag','ircot_sparse','tdca']:
    ao,an=agg(old[alg]),agg(new[alg])
    print(alg)
    for k in ['count','exact_match','soft_em','answer_f1','rougeL_f','meteor','title_hit','avg_steps','avg_llm_calls','avg_generated_tokens']:
        if k in an or k in ao:
            print(' ',k, ao.get(k), '->', an.get(k), 'delta', None if ao.get(k) is None or an.get(k) is None else round(an.get(k)-ao.get(k),6) if isinstance(an.get(k),(int,float)) else '')
print('\nSHAPE new')
for alg,p in new.items():
    rs=rows(p)
    print(alg, 'empty',sum(1 for r in rs if not (r.get('pred') or '').strip()), 'avg_retrieved', round(statistics.mean(float(r.get('num_retrieved') or 0) for r in rs),2) if alg!='tdca' else '-', 'avg_pred_words', round(statistics.mean(len((r.get('pred') or '').split()) for r in rs),2))
print('\nnew TDCA stop reasons')
from collections import Counter
nt=rowmap(new['tdca']); ot=rowmap(old['tdca']); ids=list(nt)
c=Counter(r.get('stop_reason','') for r in nt.values()); print(c)
for reason,n in c.most_common():
    vals=[r['_em'] for r in nt.values() if r.get('stop_reason','')==reason]
    print(reason, sum(vals),'/',len(vals), round(sum(vals)/len(vals),3))
print('\nTDCA changed')
for i in ids:
    if ot[i]['_em']!=nt[i]['_em'] or ot[i].get('pred')!=nt[i].get('pred') or ot[i].get('stop_reason')!=nt[i].get('stop_reason'):
        print(i, 'oldEM',ot[i]['_em'],'newEM',nt[i]['_em'],'oldstop',ot[i].get('stop_reason'),'newstop',nt[i].get('stop_reason'),'gold',nt[i]['gold'],'oldpred',ot[i].get('pred'),'newpred',nt[i].get('pred'))
print('\nnew overlap TDCA vs others')
nmaps={a:rowmap(p) for a,p in new.items()}
for alg in ['closed_book','sparse_rag','dense_rag','ircot_sparse']:
    both=sum(nmaps['tdca'][i]['_em'] and nmaps[alg][i]['_em'] for i in ids)
    tdca_only=sum(nmaps['tdca'][i]['_em'] and not nmaps[alg][i]['_em'] for i in ids)
    alg_only=sum((not nmaps['tdca'][i]['_em']) and nmaps[alg][i]['_em'] for i in ids)
    neither=sum((not nmaps['tdca'][i]['_em']) and (not nmaps[alg][i]['_em']) for i in ids)
    print('tdca vs',alg,dict(both=both,tdca_only=tdca_only,alg_only=alg_only,neither=neither))
print('\nnew IRCoT correct TDCA wrong')
for i in ids:
    if nmaps['ircot_sparse'][i]['_em'] and not nmaps['tdca'][i]['_em']:
        print(i,'gold=',nt[i]['gold'],'tdca=',nt[i]['pred'],'ircot=',nmaps['ircot_sparse'][i]['pred'])
print('\nnew TDCA correct IRCoT wrong')
for i in ids:
    if nmaps['tdca'][i]['_em'] and not nmaps['ircot_sparse'][i]['_em']:
        print(i,'gold=',nt[i]['gold'],'tdca=',nt[i]['pred'],'ircot=',nmaps['ircot_sparse'][i]['pred'])
