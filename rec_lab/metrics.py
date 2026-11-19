import numpy as np
from sklearn.metrics import roc_auc_score, log_loss, average_precision_score

def binary_metrics(y,p):
    y=np.asarray(y)
    p=np.asarray(p,dtype=float)
    if y.ndim!=1 or p.ndim!=1 or not len(y) or not set(np.unique(y)).issubset({0,1}):
        raise ValueError('Metrics need non-empty one-dimensional binary labels.')
    if not np.isfinite(p).all() or len(p)!=len(y) or (p<0).any() or (p>1).any():
        raise ValueError('Invalid predictions.')
    raw=p.copy();p=np.clip(p,1e-7,1-1e-7)
    varied=len(np.unique(y))==2
    calibration=[]
    for i in range(10):
        mask=(raw>=i/10)&(raw<(i+1)/10 if i<9 else raw<=1)
        if mask.any():
            calibration.append({'lower':i/10,'upper':(i+1)/10,'rows':int(mask.sum()),
                'mean_prediction':float(raw[mask].mean()),'positive_rate':float(y[mask].mean())})
    return {'rows':int(len(y)), 'positive_count':int(y.sum()),'positive_rate':float(y.mean()),
            'auc':float(roc_auc_score(y,p)) if varied else None,
            'logloss':float(log_loss(y,p,labels=[0,1])),
            'pr_auc_average_precision':float(average_precision_score(y,p)) if varied else None,
            'brier_score':float(np.mean((raw-y)**2)),'calibration':calibration,
            'auc_note':None if varied else 'Undefined: only one label class in this group'}

def grouped_metrics(df,predictions):
    # Group activity is measured on training only; never use test behavior counts.
    result={}
    for label,pred in predictions.items():
        groups={}
        for name,mask in [('seen_user',~df.user_unseen),('unseen_user',df.user_unseen),
                          ('seen_video',~df.video_unseen),('unseen_video',df.video_unseen)]:
            if mask.any():
                groups[name]=binary_metrics(df.loc[mask,label],np.asarray(pred)[mask.to_numpy()])
        if 'training_activity' in df:
            activity=df.training_activity
            for name,mask in [('activity_0',activity==0),('activity_1_9',(activity>=1)&(activity<10)),
                              ('activity_10_49',(activity>=10)&(activity<50)),('activity_50_plus',activity>=50)]:
                if mask.any():groups[name]=binary_metrics(df.loc[mask,label],np.asarray(pred)[mask.to_numpy()])
        result[label]=groups
    return result
