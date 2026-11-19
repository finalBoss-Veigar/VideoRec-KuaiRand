"""Inspect original logs and save reproducible data-quality summaries."""
import html
from pathlib import Path
import numpy as np
import pandas as pd
from .data import REQUIRED,write_json,sha256

def audit_logs(logs,out_dir):
    out=Path(out_dir)
    if out.exists():raise ValueError('Audit output exists; choose a new audit directory.')
    frames=[];sources=[]
    for path in logs:
        columns=list(pd.read_csv(path,nrows=0))
        missing=set(REQUIRED)-set(columns)
        if missing:raise ValueError(f'{path}: missing fields {sorted(missing)}')
        frame=pd.read_csv(path,usecols=REQUIRED)
        for c in REQUIRED:frame[c]=pd.to_numeric(frame[c],errors='coerce')
        frames.append(frame);sources.append({'name':Path(path).name,'rows':len(frame),'sha256':sha256(path)})
    data=pd.concat(frames,ignore_index=True)
    data['invalid_duration']=(~np.isfinite(data.duration_ms))|(data.duration_ms<=0)
    tabs=data.groupby('tab',dropna=False).agg(rows=('user_id','size'),users=('user_id','nunique'),
        videos=('video_id','nunique'),feedback_rate=('is_click','mean'),like_rate=('is_like','mean'),
        nonpositive_or_missing_duration=('invalid_duration','sum')).reset_index()
    date=pd.to_datetime(data.time_ms,unit='ms',utc=True,errors='coerce').dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
    daily=data.assign(day=date).groupby('day').agg(rows=('user_id','size'),feedback_rate=('is_click','mean'),like_rate=('is_like','mean')).reset_index()
    metadata={'sources':sources,'rows':len(data),'missing_or_nonnumeric':{c:int(data[c].isna().sum()) for c in REQUIRED},
              'invalid_duration_rows':int(data.invalid_duration.sum()),
              'label_values':{c:sorted(map(str,data[c].unique())) for c in ['is_click','is_like','is_rand']},
              'note':'tab values alone do not establish the UI meaning of is_click. Inspect cohort before preparing data.'}
    out.mkdir(parents=True);write_json(out/'audit.json',metadata);tabs.to_csv(out/'by_tab.csv',index=False);daily.to_csv(out/'by_day.csv',index=False)
    page='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>数据审查报告</title><style>body{font:16px/1.8 system-ui,sans-serif;margin:30px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}.wide{overflow:auto}</style><h1>数据审查报告</h1><p>原始字段审查用于帮助选择场景和处理策略。请先解释标签，再训练。</p><pre>'+html.escape(str(metadata))+'</pre><h2>按场景</h2><div class="wide">'+tabs.to_html(index=False,escape=True)+'</div><h2>按日期</h2><div class="wide">'+daily.to_html(index=False,escape=True)+'</div></html>'
    (out/'report.html').write_text(page,encoding='utf-8')
    return {'report':str(out/'report.html'),'rows':len(data),'invalid_duration_rows':metadata['invalid_duration_rows']}
