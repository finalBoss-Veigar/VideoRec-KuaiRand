"""Reproducible training and portable saved-model evaluation for v2."""
import json
import os
import platform
import random
import time
import threading
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from .data import FeatureEncoder,engineered,write_json,sha256,data_identity,validate
from .metrics import binary_metrics,grouped_metrics

def _current_rss_bytes():
    """Current process resident memory without adding an environment dependency."""
    if os.name == 'nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
                (name, ctypes.c_size_t) for name in ['PeakWorkingSetSize', 'WorkingSetSize',
                'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage']]
        counters = Counters(); counters.cb = ctypes.sizeof(counters)
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL('psapi', use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise OSError(ctypes.get_last_error(), 'GetProcessMemoryInfo failed')
        return int(counters.WorkingSetSize)
    statm = Path('/proc/self/statm')
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    return None

class ResourceMonitor:
    """Sample per-run process RSS; this includes imports and existing process state."""
    def __init__(self):
        self.peak = 0; self.samples = 0; self.error = None
        self.stop_event = threading.Event(); self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._loop, daemon=True)
    def sample(self):
        try:
            rss = _current_rss_bytes()
            if rss is not None:
                self.peak = max(self.peak, rss); self.samples += 1
        except Exception as exc:
            self.error = str(exc)
    def _loop(self):
        while not self.stop_event.wait(.05): self.sample()
    def start(self):
        self.sample(); self.thread.start(); return self
    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=1); self.sample()
        result = {'wall_seconds': time.perf_counter()-self.started,
                  'sampled_peak_process_rss_bytes': self.peak if self.samples else None,
                  'rss_samples': self.samples, 'rss_sample_interval_seconds': .05,
                  'rss_note': '50 ms sampled process RSS including libraries and prior suite state; brief peaks may be missed',
                  'rss_error': self.error}
        return result

MODELS=['constant','lr','deepfm','single','shared','mmoe']
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def now():return datetime.now(timezone.utc).isoformat()
def targets_for(task):return ['is_click','is_like'] if task=='both' else ['is_click' if task=='engage' else 'is_like']

def load_parts(data_dir,include_test=True):
    root=Path(data_dir);manifest=read_json(root/'manifest.json')
    if manifest.get('schema_version')!=2:
        raise ValueError('Use v2 prepare to create a new data version; v1 data are preserved.')
    if data_identity(manifest)!=manifest['data_id']:
        raise ValueError('Manifest protocol changed. Prepare a new data version.')
    for k in ['train','valid','test']:
        if sha256(root/f'{k}.csv')!=manifest['split_hashes'][k]:
            raise ValueError('Processed data changed. Prepare a new data version.')
    parts={k:pd.read_csv(root/f'{k}.csv') for k in (['train','valid','test'] if include_test else ['train','valid'])}
    for frame in parts.values():validate(frame)
    if parts['train'].time_ms.max()>=parts['valid'].time_ms.min():
        raise ValueError('Train/validation timestamps overlap.')
    if include_test and parts['valid'].time_ms.max()>=parts['test'].time_ms.min():
        raise ValueError('Validation/test timestamps overlap.')
    return parts,manifest

def resolve_data(root,cfg,override=None):
    choices=[Path(override)] if override else [Path(root)/cfg['data_relative_to_run'],Path(cfg['data_dir'])]
    for path in choices:
        if (path/'manifest.json').is_file():return path.resolve()
    raise FileNotFoundError('Prepared data not found. Supply evaluate --data PATH after moving a run.')

def resolve_device(requested):
    import torch
    if requested not in ['cpu','cuda','auto']:raise ValueError('device must be cpu/cuda/auto.')
    if requested=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable. Use cpu or install a matching CUDA build.')
    return 'cuda' if requested=='cuda' or requested=='auto' and torch.cuda.is_available() else 'cpu'

def train(data_dir,run_dir,model='lr',task='engage',epochs=10,batch_size=256,lr=.001,
          seed=42,like_weight=1.,variant='all',evaluate_test=False,device='cpu',
          embedding_dim=8,hidden=(32,16),experts=3,patience=3,weight_decay=1e-5,clip_grad=5.,
          aux_weight=None,selection_target=None):
    out=Path(run_dir)
    if out.exists():raise ValueError('Run directory exists. Use a new name to preserve experiments.')
    if model not in MODELS or task not in ['engage','like','both']:raise ValueError('Unknown model/task.')
    if (model in ['shared','mmoe'])!=(task=='both'):
        raise ValueError('shared/mmoe require task=both; other models are single-target.')
    if aux_weight is not None and (task!='both' or not np.isfinite(aux_weight) or aux_weight<0):
        raise ValueError('aux_weight requires task=both and a finite non-negative value.')
    if selection_target is not None and selection_target not in targets_for(task):
        raise ValueError('selection_target must be one of the trained targets.')
    if epochs<1 or batch_size<1 or like_weight<=0 or lr<=0 or patience<0 or clip_grad<0 or weight_decay<0:
        raise ValueError('Invalid training hyperparameters.')
    if not all(np.isfinite(x) for x in [like_weight,lr,weight_decay,clip_grad]):
        raise ValueError('Hyperparameters must be finite.')
    if not hidden or min(hidden)<1 or embedding_dim<1 or experts<1:
        raise ValueError('Model widths, embedding dimension and experts must be positive.')
    if device not in ['cpu','cuda','auto']:raise ValueError('Unknown device.')
    parts,manifest=load_parts(data_dir,include_test=False)
    encoder=FeatureEncoder(variant).fit(parts['train']);targets=targets_for(task)
    for target in targets:
        if parts['train'][target].nunique()<2:raise ValueError(f'{target}: training needs both classes.')
    out.mkdir(parents=True);write_json(out/'status.json',{'state':'running','started_at':now()})
    monitor=ResourceMonitor().start()
    try:
        _train(parts,manifest,encoder,targets,data_dir,out,model,task,epochs,batch_size,lr,seed,
               like_weight,variant,device,embedding_dim,hidden,experts,patience,weight_decay,clip_grad,aux_weight,selection_target)
        if evaluate_test:evaluate_saved(out)
        write_json(out/'status.json',{'state':'completed','finished_at':now()})
        return read_json(out/'metrics.json')
    except BaseException as exc:
        write_json(out/'status.json',{'state':'failed','finished_at':now(),'error':str(exc)})
        raise
    finally:
        resources=monitor.stop();write_json(out/'resources.json',resources)
        if (out/'config.json').exists():
            cfg=read_json(out/'config.json');cfg['resources']=resources;write_json(out/'config.json',cfg)

def _train(parts,manifest,encoder,targets,data_dir,out,model,task,epochs,batch_size,lr,seed,
           like_weight,variant,device,embedding_dim,hidden,experts,patience,weight_decay,clip_grad,aux_weight,selection_target):
    random.seed(seed);np.random.seed(seed)
    cfg={'schema_version':2,'model':model,'task':task,'epochs':epochs,'batch_size':batch_size,
         'learning_rate':lr,'seed':seed,'like_weight':like_weight,'aux_weight':aux_weight,
         'selection_target':selection_target,'features':variant,
         'requested_device':device,'device':'cpu','patience':patience,'weight_decay':weight_decay,'clip_grad':clip_grad,
         'data_dir':str(Path(data_dir).resolve()),'data_relative_to_run':os.path.relpath(Path(data_dir).resolve(),out.resolve()),
         'dataset_kind':manifest['dataset_kind'],'data_id':manifest['data_id'],'data_hashes':manifest['split_hashes'],
         'selection':f'lowest validation {selection_target} BCE' if selection_target else 'lowest weighted validation BCE',
         'loss_formula':'BCE_like + aux_weight * BCE_click' if aux_weight is not None else 'legacy mean of weighted task BCE',
         'test_evaluated':False,'model_note':'DeepFM categorical FM; numeric fields in linear/MLP branches',
         'architecture':{'embedding_dim':embedding_dim,'hidden':list(hidden),'experts':experts},
         'lr_baseline':{'C':1.,'max_iter':300},
         'source_hashes':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}}
    versions={'python':platform.python_version(),'platform':platform.platform(),'numpy':np.__version__,
              'pandas':pd.__version__,'sklearn':sklearn.__version__}
    write_json(out/'config.json',cfg);write_json(out/'data_manifest.json',manifest)
    write_json(out/'encoder.json',encoder.to_dict())
    start=time.perf_counter();history=[]
    if model=='constant':
        probability=float(parts['train'][targets[0]].mean());write_json(out/'constant.json',{'probability':probability})
        predict=lambda frame:np.full((len(frame),1),probability)
        history=[{'epoch':0,'valid_loss':binary_metrics(parts['valid'][targets[0]],predict(parts['valid'])[:,0])['logloss']}]
        cfg['parameter_count']=1;cfg['best_epoch']=0
    elif model=='lr':
        transform=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore'),encoder.categories),
                                     ('num',StandardScaler(),encoder.numbers)])
        estimator=Pipeline([('features',transform),('model',LogisticRegression(max_iter=300,C=1,random_state=seed))])
        estimator.fit(engineered(parts['train']),parts['train'][targets[0]])
        import pickle
        with open(out/'model.pkl','wb') as f:pickle.dump(estimator,f)
        predict=lambda frame:estimator.predict_proba(engineered(frame))[:,1:2]
        history=[{'epoch':0,'valid_loss':binary_metrics(parts['valid'][targets[0]],predict(parts['valid'])[:,0])['logloss']}]
        cfg['parameter_count']=int(estimator.named_steps['model'].coef_.size+1);cfg['best_epoch']=0
    else:
        import torch
        from torch.utils.data import DataLoader,TensorDataset
        from .models import build_model
        torch.manual_seed(seed);torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        cfg['device']=resolve_device(device);versions['torch']=torch.__version__;versions['cuda']=torch.version.cuda
        net=build_model(model,encoder.sizes,len(encoder.numbers),embedding_dim,tuple(hidden),experts).to(cfg['device'])
        cfg['parameter_count']=sum(p.numel() for p in net.parameters())
        if cfg['device']=='cuda':torch.cuda.reset_peak_memory_stats()
        optimizer=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay)
        loss_fn=torch.nn.BCEWithLogitsLoss(reduction='none')
        selected=[0,1] if task=='both' else [0 if task=='engage' else 1]
        # Task order remains [is_click, is_like]. New auxiliary weighting leaves main-task strength fixed.
        weights=torch.tensor([aux_weight,1.] if aux_weight is not None else ([1.,like_weight] if task=='both' else [1.]),device=cfg['device'])
        loss_scale=2. if aux_weight is not None else 1.
        selection_index=targets.index(selection_target) if selection_target else None
        c,n,y=encoder.transform(parts['train'])
        dataset=TensorDataset(torch.tensor(c),torch.tensor(n),torch.tensor(y[:,selected]))
        loader=DataLoader(dataset,batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed),num_workers=0)
        vc,vn,vy=encoder.transform(parts['valid']);vc,vn,vy=torch.tensor(vc),torch.tensor(vn),torch.tensor(vy[:,selected])
        best=float('inf');best_state=None;stale=0
        for epoch in range(1,epochs+1):
            net.train();total=0.
            for bc,bn,by in loader:
                bc,bn,by=(v.to(cfg['device']) for v in [bc,bn,by]);optimizer.zero_grad(set_to_none=True)
                loss=(loss_fn(net(bc,bn),by)*weights).mean()*loss_scale
                if not torch.isfinite(loss):raise ValueError('Non-finite loss; inspect data and learning rate.')
                loss.backward()
                if clip_grad:torch.nn.utils.clip_grad_norm_(net.parameters(),clip_grad)
                optimizer.step();total+=float(loss.detach())*len(bc)
            net.eval();valid_total=0.;valid_by_target=np.zeros(len(targets),dtype=float)
            with torch.no_grad():
                for a in range(0,len(vc),batch_size):
                    bc,bn,by=(v[a:a+batch_size].to(cfg['device']) for v in [vc,vn,vy])
                    task_losses=loss_fn(net(bc,bn),by)
                    valid_by_target+=task_losses.sum(dim=0).cpu().numpy()
                    selected_loss=task_losses[:,selection_index].mean() if selection_index is not None else (task_losses*weights).mean()*loss_scale
                    valid_total+=float(selected_loss)*len(bc)
            value=valid_total/len(vc)
            if not np.isfinite(value):raise ValueError('Non-finite validation loss.')
            history.append({'epoch':epoch,'train_loss':total/len(dataset),'valid_loss':value,
                **{f'valid_{target}_loss':float(valid_by_target[j]/len(vc)) for j,target in enumerate(targets)}})
            if value<best-1e-8:
                best=value;stale=0;cfg['best_epoch']=epoch
                best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
            else:stale+=1
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            print(f'[{out.name}] epoch={epoch} train={history[-1]["train_loss"]:.5f} valid={value:.5f}',flush=True)
            if patience and stale>=patience:break
        net.load_state_dict(best_state);torch.save(best_state,out/'model.pt')
        cfg['cuda_peak_allocated_bytes']=int(torch.cuda.max_memory_allocated()) if cfg['device']=='cuda' else None
        cfg['cuda_peak_reserved_bytes']=int(torch.cuda.max_memory_reserved()) if cfg['device']=='cuda' else None
        def predict(frame):
            c,n=encoder.encode(frame);chunks=[];net.eval()
            with torch.no_grad():
                for a in range(0,len(c),batch_size):
                    bc=torch.tensor(c[a:a+batch_size],device=cfg['device']);bn=torch.tensor(n[a:a+batch_size],device=cfg['device'])
                    chunks.append(net(bc,bn).sigmoid().cpu().numpy())
            return np.concatenate(chunks)
    cfg['epochs_completed']=len(history) if model not in ['constant','lr'] else 0;cfg['training_seconds']=time.perf_counter()-start
    write_json(out/'config.json',cfg);write_json(out/'versions.json',versions);pd.DataFrame(history).to_csv(out/'history.csv',index=False)
    metrics={'valid':save_evaluation(out,'valid',parts['valid'],predict(parts['valid']),targets,parts['train']),
             'training_seconds':cfg['training_seconds'],'dataset_kind':manifest['dataset_kind']}
    write_json(out/'metrics.json',metrics)

def save_evaluation(root,split,frame,probs,targets,train_frame):
    frame=frame.copy();frame['user_unseen']=~frame.user_id.isin(train_frame.user_id)
    frame['video_unseen']=~frame.video_id.isin(train_frame.video_id)
    frame['training_activity']=frame.user_id.map(train_frame.user_id.value_counts()).fillna(0).astype(int)
    predictions={target:probs[:,j] for j,target in enumerate(targets)}
    metrics={target:binary_metrics(frame[target],predictions[target]) for target in targets}
    write_json(root/f'{split}_groups.json',grouped_metrics(frame,predictions))
    output=frame[(['source_row_id'] if 'source_row_id' in frame else [])+['user_id','video_id','time_ms','tab']+targets].copy()
    for target,p in predictions.items():output[f'p_{target}']=p
    output.to_csv(root/f'{split}_predictions.csv',index=False);return metrics

def saved_probabilities(run_dir,frame,device='cpu'):
    root=Path(run_dir);cfg=read_json(root/'config.json');enc=FeatureEncoder.from_dict(read_json(root/'encoder.json'))
    cat,num=enc.encode(frame)
    if cfg['model']=='constant':return np.full((len(frame),1),read_json(root/'constant.json')['probability'])
    if cfg['model']=='lr':
        import pickle
        with open(root/'model.pkl','rb') as f:estimator=pickle.load(f)
        return estimator.predict_proba(engineered(frame))[:,1:2]
    import torch
    from .models import build_model
    selected=resolve_device(device);a=cfg['architecture']
    net=build_model(cfg['model'],enc.sizes,len(enc.numbers),a['embedding_dim'],tuple(a['hidden']),a['experts'])
    net.load_state_dict(torch.load(root/'model.pt',map_location='cpu',weights_only=True));net.to(selected);net.eval()
    results=[];batch=cfg['batch_size']
    with torch.no_grad():
        for i in range(0,len(cat),batch):
            results.append(net(torch.tensor(cat[i:i+batch],device=selected),torch.tensor(num[i:i+batch],device=selected)).sigmoid().cpu().numpy())
    return np.concatenate(results)

def predict_saved(run_dir,input_csv,output_csv,device='cpu'):
    if Path(output_csv).exists():raise ValueError('Prediction output exists; choose a new path.')
    frame=pd.read_csv(input_csv);cfg=read_json(Path(run_dir)/'config.json');probabilities=saved_probabilities(run_dir,frame,device)
    output=frame[(['source_row_id'] if 'source_row_id' in frame else [])+['user_id','video_id','time_ms','tab','duration_ms']].copy()
    for j,target in enumerate(targets_for(cfg['task'])):output[f'p_{target}']=probabilities[:,j]
    Path(output_csv).parent.mkdir(parents=True,exist_ok=True);output.to_csv(output_csv,index=False)
    return {'rows':len(output),'output':str(output_csv)}

def evaluate_saved(run_dir,split='test',data_dir=None):
    if split!='test':raise ValueError('Final evaluation supports split=test.')
    root=Path(run_dir);cfg=read_json(root/'config.json');metrics=read_json(root/'metrics.json')
    if split in metrics:raise ValueError('Test already evaluated. Preserve results; never tune repeatedly against test.')
    data=resolve_data(root,cfg,data_dir);parts,manifest=load_parts(data,include_test=True)
    if manifest['data_id']!=cfg['data_id']:raise ValueError('Data protocol differs from model training.')
    probs=saved_probabilities(root,parts[split])
    metrics[split]=save_evaluation(root,split,parts[split],probs,targets_for(cfg['task']),parts['train'])
    metrics['test_evaluated_at']=now();cfg['test_evaluated']=True
    write_json(root/'metrics.json',metrics);write_json(root/'config.json',cfg);return metrics[split]

def compare(run_dirs,output):
    rows=[];identities=set()
    for root in map(Path,run_dirs):
        if read_json(root/'status.json')['state']!='completed':raise ValueError('Only completed runs can be compared.')
        cfg=read_json(root/'config.json');m=read_json(root/'metrics.json');identities.add(cfg['data_id'])
        for split in ['valid','test']:
            for target,value in m.get(split,{}).items():
                rows.append({'run':root.name,'model':cfg['model'],'task':cfg['task'],'seed':cfg['seed'],
                    'like_weight':cfg['like_weight'],'aux_weight':cfg.get('aux_weight'),'selection_target':cfg.get('selection_target'),
                    'features':cfg['features'],'dataset_kind':cfg['dataset_kind'],
                    'data_id':cfg['data_id'],'split':split,'target':target,'split_hash':cfg['data_hashes'][split],
                    'embedding_dim':cfg['architecture']['embedding_dim'],'hidden':str(cfg['architecture']['hidden']),
                    'experts':cfg['architecture']['experts'],'batch_size':cfg['batch_size'],
                    'learning_rate':cfg['learning_rate'],'weight_decay':cfg['weight_decay'],'patience':cfg['patience'],
                    'epoch_budget':cfg['epochs'],'device':cfg['device'],**{k:v for k,v in value.items() if k!='calibration'},
                    'parameters':cfg['parameter_count'],'training_seconds':cfg['training_seconds'],
                    'sampled_peak_process_rss_bytes':cfg.get('resources',{}).get('sampled_peak_process_rss_bytes'),
                    'cuda_peak_allocated_bytes':cfg.get('cuda_peak_allocated_bytes')})
    if len(identities)!=1:raise ValueError('Comparisons require identical data and protocol identity.')
    result=pd.DataFrame(rows)
    if result.empty:raise ValueError('No metrics to compare.')
    destination=Path(output)
    if destination.exists():raise ValueError('Comparison output exists; use a new path.')
    destination.parent.mkdir(parents=True,exist_ok=True);result.to_csv(destination,index=False);return result
