import argparse
import json
import sys
import pandas as pd
from .data import prepare,make_demo,audit

def main():
    parser=argparse.ArgumentParser(description='Offline video feedback prediction teaching lab')
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('demo');p.add_argument('--out',default='data/raw/synthetic.csv');p.add_argument('--rows',type=int,default=6000);p.add_argument('--seed',type=int,default=42)
    sub.add_parser('doctor')
    p=sub.add_parser('suite');p.add_argument('--config',required=True);p.add_argument('--data');p.add_argument('--out')
    p=sub.add_parser('report');p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('inspect');p.add_argument('--csv',required=True)
    p=sub.add_parser('audit');p.add_argument('--logs',nargs='+',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('prepare');p.add_argument('--logs',nargs='+',required=True);p.add_argument('--out',required=True);p.add_argument('--kind',choices=['synthetic','real'],required=True);p.add_argument('--tab',type=int);p.add_argument('--policy',choices=['standard','random'],default='standard');p.add_argument('--signal-note',default='');p.add_argument('--max-rows',type=int,default=0)
    p.add_argument('--invalid-rows',choices=['error','drop'],default='error');p.add_argument('--train-before');p.add_argument('--valid-before')
    p=sub.add_parser('train');p.add_argument('--data',required=True);p.add_argument('--run',required=True);p.add_argument('--model',choices=['constant','lr','deepfm','single','shared','mmoe'],default='lr');p.add_argument('--task',choices=['engage','like','both'],default='engage');p.add_argument('--epochs',type=int,default=5);p.add_argument('--batch-size',type=int,default=256);p.add_argument('--lr',type=float,default=.001);p.add_argument('--seed',type=int,default=42);p.add_argument('--like-weight',type=float,default=1.);p.add_argument('--features',choices=['all','no_user','no_video','context'],default='all');p.add_argument('--evaluate-test',action='store_true')
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu');p.add_argument('--embedding-dim',type=int,default=8);p.add_argument('--hidden',nargs='+',type=int,default=[32,16]);p.add_argument('--experts',type=int,default=3);p.add_argument('--patience',type=int,default=3);p.add_argument('--weight-decay',type=float,default=1e-5);p.add_argument('--clip-grad',type=float,default=5.)
    p.add_argument('--aux-weight',type=float);p.add_argument('--selection-target',choices=['is_click','is_like'])
    p=sub.add_parser('predict');p.add_argument('--run',required=True);p.add_argument('--input',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    p=sub.add_parser('compare');p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True)
    p=sub.add_parser('evaluate');p.add_argument('--run',required=True);p.add_argument('--split',choices=['test'],default='test')
    p.add_argument('--data')
    args=parser.parse_args()
    if args.command=='doctor':
        from .doctor import doctor
        result=doctor()
    elif args.command=='suite':
        from .experiments import suite
        result=suite(args.config,args.data,args.out)
    elif args.command=='report':
        from .report import report
        result=report(args.runs,args.out)
    elif args.command=='audit':
        from .audit import audit_logs
        result=audit_logs(args.logs,args.out)
    elif args.command=='demo':
        result=make_demo(args.out,args.rows,args.seed)
    elif args.command=='inspect':
        df=pd.read_csv(args.csv)
        result={'columns':list(df),'head':df.head(3).to_dict(orient='records'),'rows':len(df),
                'tabs':{str(k):int(v) for k,v in df.tab.value_counts().items()} if 'tab' in df else {},
                'label_by_tab':df.groupby('tab')[['is_click','is_like']].agg(['mean','sum','count']).to_string() if {'tab','is_click','is_like'}.issubset(df) else ''}
    elif args.command=='prepare':
        result=prepare(args.logs,args.out,args.kind,args.tab,args.policy,args.signal_note,args.max_rows,args.invalid_rows,args.train_before,args.valid_before)
    else:
        from .runner import train,predict_saved,compare,evaluate_saved
        if args.command=='train':
            result=train(args.data,args.run,args.model,args.task,args.epochs,args.batch_size,args.lr,args.seed,args.like_weight,args.features,args.evaluate_test,args.device,args.embedding_dim,args.hidden,args.experts,args.patience,args.weight_decay,args.clip_grad,args.aux_weight,args.selection_target)
        elif args.command=='predict':
            result=predict_saved(args.run,args.input,args.out,args.device)
        elif args.command=='evaluate':
            result=evaluate_saved(args.run,args.split,args.data)
        else:
            result=compare(args.runs,args.out).to_dict(orient='records')
    print(json.dumps(result,ensure_ascii=False,indent=2,default=str))

if __name__=='__main__':
    try:
        main()
    except (ValueError,FileNotFoundError,KeyError,ImportError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        sys.exit(2)
