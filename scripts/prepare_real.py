"""Convenience wrapper for the documented KuaiRand-Pure v2 protocol."""
import argparse
import json
from pathlib import Path
from rec_lab.data import prepare

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--tab',type=int,required=True)
    p.add_argument('--raw',default='data/raw/kuairand_extracted/KuaiRand-Pure/data')
    p.add_argument('--out',default='data/processed/real_v2')
    p.add_argument('--max-rows',type=int,default=50000)
    p.add_argument('--drop-invalid',action='store_true')
    p.add_argument('--signal-note',default='tab=1 original is_click feedback and is_like; no uniform click-only UI claim; offline standard-exposure sample')
    a=p.parse_args()
    if a.tab!=1 and a.signal_note==p.get_default('signal_note'):
        p.error('For a different tab, supply your own --signal-note.')
    raw=Path(a.raw)
    logs=[raw/'log_standard_4_08_to_4_21_pure.csv',raw/'log_standard_4_22_to_5_08_pure.csv']
    m=prepare(logs,a.out,'real',a.tab,'standard',a.signal_note,a.max_rows,'drop' if a.drop_invalid else 'error')
    print(json.dumps({'data_id':m['data_id'],'output':a.out,'removed':m['dropped_invalid_rows'],
                      'splits':{k:v['rows'] for k,v in m['split_audit'].items()}},indent=2))

if __name__=='__main__':main()
