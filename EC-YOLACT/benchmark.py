"""Batch-one FP32 timing with device synchronization; excludes I/O/preprocess."""
import argparse,json,time,platform
from pathlib import Path
import numpy as np,torch
from experiment import load_trained
from evaluate import input_tensor
from utils.anchors import get_anchors
from utils.utils_bbox import BBoxUtility

def main():
    p=argparse.ArgumentParser(); p.add_argument('--weights',required=True); p.add_argument('--image',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--warmup',type=int,default=30); p.add_argument('--runs',type=int,default=100); p.add_argument('--output',default='benchmark.json'); a=p.parse_args()
    if a.runs<1 or a.warmup<0: raise ValueError('Invalid repetition count')
    device=torch.device(a.device); model,ck=load_trained(a.weights,device); c=ck['config']; x,shape=input_tensor(a.image,c['input_shape'],device)
    anchors=torch.tensor(get_anchors(c['input_shape'],c['anchor_sizes']),dtype=torch.float32,device=device); bbox=BBoxUtility()
    def sync():
        if device.type=='cuda': torch.cuda.synchronize(device)
    report={'batch':1,'precision':'FP32','torch':str(torch.__version__),'python':platform.python_version(),'device':torch.cuda.get_device_name(device) if device.type=='cuda' else platform.processor(),'shape':c['input_shape'],'variant':ck['variant'],'warmup':a.warmup,'runs':a.runs,'scope':'Excludes loading, preprocessing, device transfer and visualization'}
    with torch.inference_mode():
        for post in (False,True):
            times=[]
            for i in range(a.warmup+a.runs):
                sync(); t=time.perf_counter(); pred=model(x)
                if post: bbox.decode_nms(pred,anchors,c['confidence'],c['nms_iou'],shape,max_detections=100)
                sync()
                if i>=a.warmup: times.append(time.perf_counter()-t)
            report['forward_postprocess' if post else 'forward']={'mean_ms':float(np.mean(times)*1000),'median_ms':float(np.median(times)*1000),'FPS':float(1/np.mean(times))}
    Path(a.output).write_text(json.dumps(report,indent=2),encoding='utf-8'); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
