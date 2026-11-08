"""Synthetic end-to-end smoke test; metrics are NOT wafer experiment results."""
import json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
root=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as td:
 p=Path(td); c=json.loads((root/'configs/wsd.json').read_text()); c.update(backbone='resnet50',epochs=1,batch_size=2,workers=0,input_shape=[64,64])
 cats=[{'id':i+1,'name':n} for i,n in enumerate(c['classes'])]
 for split in ('train','val'):
  images=[]; anns=[]
  for i in (1,2):
   name=f'{split}{i}.png'; image=np.zeros((64,64,3),dtype=np.uint8); image[16:40,16:40]=180; Image.fromarray(image).save(p/name)
   images.append({'id':i,'file_name':name,'height':64,'width':64})
   anns.append({'id':i,'image_id':i,'category_id':1,'bbox':[16,16,24,24],'area':576,'iscrowd':0,'segmentation':[[16,16,40,16,40,40,16,40]]})
  path=p/f'{split}.json'; path.write_text(json.dumps({'images':images,'annotations':anns,'categories':cats,'info':{}}))
  c[f'{split}_images']=str(p);c[f'{split}_annotations']=str(path)
 cfg=p/'config.json';cfg.write_text(json.dumps(c)); run=p/'run'; env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 cmds=[['train.py','--config',str(cfg),'--variant','ec','--scratch','--device','cpu','--output',str(run)],['evaluate.py','--weights',str(run/'best_loss.pt'),'--device','cpu','--output',str(p/'eval')],['benchmark.py','--weights',str(run/'best_loss.pt'),'--image',str(p/'val1.png'),'--device','cpu','--warmup','1','--runs','2','--output',str(p/'bench.json')]]
 cmds.append(['predict.py','--weights',str(run/'best_loss.pt'),'--image',str(p/'val1.png'),'--device','cpu','--output',str(p/'predict')])
 for cmd in cmds: subprocess.run([sys.executable,*cmd],cwd=root,env=env,check=True)
 assert (run/'training.csv').is_file() and (p/'eval/metrics.csv').is_file()
 print('PASS synthetic training -> strict checkpoint reload -> bbox/mask evaluation -> timing. No WSD performance claim.')
