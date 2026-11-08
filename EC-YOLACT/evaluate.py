"""COCO bbox/segm evaluation on the complete validation split."""
import argparse,csv,json
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from experiment import *
from utils.anchors import get_anchors
from utils.utils import preprocess_input
from utils.utils_bbox import BBoxUtility
from utils.utils_map import Make_json,prep_metrics

def input_tensor(path,shape,device):
    im=np.asarray(Image.open(path).convert('RGB'),dtype=np.float32)
    x=preprocess_input(cv2.resize(im,(shape[1],shape[0]))).transpose(2,0,1)
    return torch.from_numpy(x.copy()).float().unsqueeze(0).to(device),im.shape[:2]

def score(co,rows,kind):
    if rows: dt=co.loadRes(rows)
    else:
        dt=COCO(); dt.dataset={'images':list(co.imgs.values()),'categories':list(co.cats.values()),'annotations':[]}; dt.createIndex()
    ev=COCOeval(co,dt,kind); ev.params.imgIds=sorted(co.getImgIds()); ev.evaluate(); ev.accumulate(); ev.summarize()
    names=['AP','AP50','AP75','APS','APM','APL','AR1','AR10','AR100','ARS','ARM','ARL']
    result={k:float(v*100) if v>=0 else None for k,v in zip(names,ev.stats)}
    r=ev.eval['recall'][0,:,0,2]; r=r[r>=0]; result['Recall50']=float(r.mean()*100) if len(r) else None
    per={}
    for k,cid in enumerate(ev.params.catIds):
        p=ev.eval['precision'][0,:,k,0,2]; p=p[p>=0]
        per[co.cats[cid]['name']]=float(p.mean()*100) if len(p) else None
    return result,per

def main():
    p=argparse.ArgumentParser(); p.add_argument('--weights',required=True); p.add_argument('--config'); p.add_argument('--device',default='cuda'); p.add_argument('--output',required=True)
    a=p.parse_args(); device=torch.device(a.device); model,ck=load_trained(a.weights,device); c=read_config(a.config) if a.config else ck['config']
    for k in ('classes','input_shape','anchor_sizes','backbone','ca_normalization'):
        if c[k]!=ck['config'][k]: raise ValueError(f'Checkpoint/config mismatch: {k}')
    check_data(c); co,labels=get_data(c,'val'); co.dataset.setdefault('info',{})
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True); mj=Make_json(str(out),labels)
    anchors=torch.tensor(get_anchors(c['input_shape'],c['anchor_sizes']),dtype=torch.float32,device=device); bbox=BBoxUtility()
    with torch.inference_mode():
        for index,iid in enumerate(sorted(co.getImgIds())):
            x,shape=input_tensor(Path(c['val_images'])/co.imgs[iid]['file_name'],c['input_shape'],device)
            boxes,scores,classes,_,masks=bbox.decode_nms(model(x),anchors,c['confidence'],c['nms_iou'],shape,max_detections=100)
            if boxes is not None: prep_metrics(boxes.cpu().numpy(),scores.cpu().numpy(),classes.cpu().numpy(),masks.cpu().numpy(),iid,mj)
            if (index+1)%100==0: print(f'{index+1}/{len(co.imgs)}',flush=True)
    mj.dump(); metrics={}; per={}
    for kind,rows in [('bbox',mj.bbox_data),('segm',mj.mask_data)]: metrics[kind],per[kind]=score(co,rows,kind)
    (out/'metrics.json').write_text(json.dumps({'metrics_percent':metrics,'per_class_AP50':per,'config':c,'variant':ck['variant'],'checkpoint':str(Path(a.weights).resolve()),'epoch':ck['epoch']},indent=2),encoding='utf-8')
    with (out/'metrics.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=['type',*metrics['bbox']]); wr.writeheader()
        for kind,row in metrics.items(): wr.writerow({'type':kind,**row})
    with (out/'per_class.csv').open('w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['class','bbox_AP50','mask_AP50'])
        for name in c['classes']: wr.writerow([name,per['bbox'][name],per['segm'][name]])
if __name__=='__main__': main()
