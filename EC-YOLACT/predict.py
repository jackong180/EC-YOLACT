"""Save a qualitative overlay and individual binary masks for one image."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from experiment import load_trained
from evaluate import input_tensor
from utils.anchors import get_anchors
from utils.utils_bbox import BBoxUtility

def main():
 p=argparse.ArgumentParser(); p.add_argument('--weights',required=True); p.add_argument('--image',required=True); p.add_argument('--output',required=True); p.add_argument('--device',default='cuda'); p.add_argument('--confidence',type=float,default=.3);a=p.parse_args()
 device=torch.device(a.device); model,ck=load_trained(a.weights,device); c=ck['config']; x,shape=input_tensor(a.image,c['input_shape'],device)
 anchors=torch.tensor(get_anchors(c['input_shape'],c['anchor_sizes']),dtype=torch.float32,device=device)
 with torch.inference_mode(): boxes,scores,classes,_,masks=BBoxUtility().decode_nms(model(x),anchors,a.confidence,c['nms_iou'],shape)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True); im=Image.open(a.image).convert('RGB'); arr=np.array(im); records=[]
 if boxes is not None:
  boxes=boxes.cpu().numpy(); scores=scores.cpu().numpy(); classes=classes.cpu().numpy(); masks=masks.cpu().numpy()
  colors=np.random.default_rng(11).integers(40,240,(len(c['classes']),3))
  for i,cls in enumerate(classes):
   color=colors[int(cls)]; mask=masks[:,:,i];arr[mask]=(.55*arr[mask]+.45*color).astype(np.uint8)
   Image.fromarray((mask*255).astype(np.uint8)).save(out/f'mask_{i:03d}.png')
   records.append({'class':c['classes'][int(cls)],'score':float(scores[i]),'xyxy':boxes[i].tolist(),'mask':f'mask_{i:03d}.png'})
  im=Image.fromarray(arr);draw=ImageDraw.Draw(im)
  for rec in records:
   b=rec['xyxy'];draw.rectangle(b,outline=(255,60,60),width=2);draw.text((b[0],max(0,b[1]-12)),f'{rec["class"]} {rec["score"]:.2f}',fill=(255,40,40))
 im.save(out/'overlay.png');(out/'instances.json').write_text(json.dumps(records,indent=2))
 print(f'Saved {len(records)} instances to {out}')
if __name__=='__main__':main()
