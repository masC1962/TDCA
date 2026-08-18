import csv,json,statistics
from pathlib import Path
runs={
 'closed_book': Path('outputs/musique_closed_book_qwen_api_20260520_095813_closed_book_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'sparse_rag': Path('outputs/musique_sparse_rag_qwen_api_20260520_095848_sparse_rag_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'dense_rag': Path('outputs/musique_dense_rag_qwen_api_20260520_095930_dense_rag_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'ircot_sparse': Path('outputs/musique_ircot_sparse_qwen_api_20260520_100334_ircot_qwen-plus_tok1200_n50_musique_qwen_tok1200_n50'),
 'tdca': Path('batch_outputs/musique_ans_v1.0_dev_20260520_100704'),
}
def rows(p):
    with open(p/'summary.csv',encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))
for alg,p in runs.items():
    rs=rows(p)
    empty=sum(1 for r in rs if not (r.get('pred') or '').strip())
    em=sum(int(float(r.get('exact_match') or 0)) for r in rs)
    soft=sum(int(float(r.get('soft_em') or 0)) for r in rs)
    f1=[float(r.get('answer_f1') or 0) for r in rs]
    pred_tok=[len((r.get('pred') or '').split()) for r in rs]
    partial=sum(1 for r in rs if not int(float(r.get('exact_match') or 0)) and float(r.get('answer_f1') or 0)>=0.5)
    print('\n',alg)
    print('empty',empty,'em',em,'soft',soft,'partial_wrong_f1>=.5',partial,'avg_pred_words',round(statistics.mean(pred_tok),2),'median_pred_words',statistics.median(pred_tok))
    if alg=='tdca':
        print('avg steps', statistics.mean(float(r.get('steps') or 0) for r in rs), 'avg calls', statistics.mean(float(r.get('llm_calls') or 0) for r in rs), 'avg tok', statistics.mean(float(r.get('generated_tokens') or 0) for r in rs))
        print('empty ids',[r['sample_id'] for r in rs if not (r.get('pred') or '').strip()])
    else:
        print('avg retrieved', statistics.mean(float(r.get('num_retrieved') or 0) for r in rs))
