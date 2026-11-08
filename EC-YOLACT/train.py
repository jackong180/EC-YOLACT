"""Run from repository root. No implicit LR scaling or pretrained downloads."""
import argparse,csv,json,math,platform,time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from experiment import *
from utils.anchors import get_anchors
from utils.augmentations import Augmentation
from utils.dataloader import COCODetection,dataset_collate
from nets.yolact_training import Multi_Loss

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/wsd.json'); ap.add_argument('--variant',choices=ABLATIONS,default='ec')
    g=ap.add_mutually_exclusive_group(required=True); g.add_argument('--init'); g.add_argument('--scratch',action='store_true')
    ap.add_argument('--device',default='cuda'); ap.add_argument('--output',required=True); ap.add_argument('--batch-size',type=int); ap.add_argument('--workers',type=int)
    a=ap.parse_args(); c=read_config(a.config)
    if a.batch_size: c['batch_size']=a.batch_size
    if a.workers is not None: c['workers']=a.workers
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'last.pt').exists(): raise FileExistsError('Use a fresh output folder; existing run will not be overwritten')
    report=check_data(c); seed_all(c['seed']); device=torch.device(a.device)
    model=make_model(c,a.variant).to(device)
    init=load_init(model,a.init) if a.init else {'initialization':'scratch'}
    (out/'initialization.json').write_text(json.dumps(init,indent=2),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    (out/'environment.json').write_text(json.dumps({'python':platform.python_version(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'device':str(device),'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,'data':report},indent=2))
    loaders=[]
    for split in ('train','val'):
        co,labels=get_data(c,split)
        ds=COCODetection(c[f'{split}_images'],co,labels,Augmentation(c['input_shape']) if split=='train' else None,c['input_shape'])
        loaders.append(DataLoader(ds,batch_size=c['batch_size'],shuffle=split=='train',num_workers=c['workers'],collate_fn=dataset_collate,worker_init_fn=seed_worker,generator=torch.Generator().manual_seed(c['seed']),drop_last=False,pin_memory=device.type=='cuda'))
    anchors=torch.tensor(get_anchors(c['input_shape'],c['anchor_sizes']),dtype=torch.float32,device=device)
    loss_fn=Multi_Loss(len(c['classes'])+1,anchors,.5,.4,3)
    opt=torch.optim.SGD(model.parameters(),lr=c['lr'],momentum=c['momentum'],weight_decay=c['weight_decay'],nesterov=True)
    best=float('inf')
    with (out/'training.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['epoch','lr','train_loss','val_loss','seconds'])
        for epoch in range(c['epochs']):
            lr=c['min_lr']+(c['lr']-c['min_lr'])*(1+math.cos(math.pi*epoch/max(1,c['epochs']-1)))/2
            for pg in opt.param_groups: pg['lr']=lr
            start=time.perf_counter(); means=[]
            for phase,loader in enumerate(loaders):
                model.train(phase==0); total=0.; count=0
                for images,targets,masks,crowds in loader:
                    images=images.to(device); targets=[t.to(device) for t in targets]; masks=[m.to(device) for m in masks]
                    with torch.set_grad_enabled(phase==0):
                        loss=sum(loss_fn(model(images),targets,masks,crowds).values())
                        if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss at epoch {epoch+1}')
                        if phase==0: opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    total+=loss.item()*len(images); count+=len(images)
                means.append(total/count)
            state={'model_state':model.state_dict(),'optimizer':opt.state_dict(),'config':c,'variant':a.variant,'epoch':epoch+1,'val_loss':means[1]}
            torch.save(state,out/'last.pt')
            if means[1]<best: best=means[1]; torch.save(state,out/'best_loss.pt')
            writer.writerow([epoch+1,lr,*means,time.perf_counter()-start]); f.flush()
            print(f'Epoch {epoch+1}/{c["epochs"]} train={means[0]:.5f} val={means[1]:.5f}',flush=True)
if __name__=='__main__': main()
