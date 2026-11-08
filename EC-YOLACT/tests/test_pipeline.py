import unittest,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from PIL import Image
from experiment import *
from utils.EGA import EnergyAttention
from utils.CA import CausalAttention
from utils.anchors import get_anchors
from utils.dataloader import COCODetection
from nets.yolact_training import Multi_Loss
from evaluate import score
from utils.utils_bbox import BBoxUtility
torch.set_num_threads(2)
class TestPipeline(unittest.TestCase):
 def test_attention(self):
  x=torch.randn(2,16,8,8,requires_grad=True)
  for layer in [EnergyAttention(),CausalAttention(16),CausalAttention(16,normalization='mean')]:
   self.assertTrue(torch.equal(layer(x),x)); layer.gamma.data.fill_(1); y=layer(x)
   self.assertTrue(torch.isfinite(y).all()); y.square().mean().backward(); self.assertIsNotNone(layer.gamma.grad)
  ega=EnergyAttention(); ega.gamma.data.fill_(.5); e=x.square().sum(1,keepdim=True)
  torch.testing.assert_close(ega(x),x+.5*x*e/(e.sum((2,3),keepdim=True)+1e-6))
  self.assertTrue(torch.isfinite(ega(torch.zeros_like(x))).all())
 def test_six_variants_and_r101(self):
  c=read_config('configs/wsd.json'); c['backbone']='resnet50'
  for v,(b,p) in ABLATIONS.items():
   m=make_model(c,v,False).eval()
   self.assertEqual(sum(not isinstance(l,torch.nn.Identity) for l in m.backbone.causal_attention),0 if b=='none' else 4)
   self.assertEqual(isinstance(m.proto_attention,torch.nn.Identity),p=='none')
   with torch.no_grad(): out=m(torch.randn(1,3,64,64))
   self.assertEqual(out[0].shape[1],len(get_anchors([64,64]))); self.assertEqual(out[3].shape,(1,16,16,32))
  c['backbone']='resnet101'; m=make_model(c,'ec',False).eval(); self.assertEqual(len(m.backbone.layers[2]),23)
  with torch.no_grad(): out=m(torch.randn(1,3,544,544))
  self.assertEqual(out[0].shape,(1,18525,4)); self.assertEqual(out[3].shape,(1,136,136,32))
 def test_loss_backward(self):
  c=read_config('configs/wsd.json'); c['backbone']='resnet50'; m=make_model(c,'ec').eval()
  anchors=torch.tensor(get_anchors([64,64]),dtype=torch.float32); crit=Multi_Loss(8,anchors,.5,.4,3)
  for crowd,empty in [(0,False),(0,True),(1,False)]:
   targets=[torch.zeros(0,5) if empty else torch.tensor([[.2,.2,.7,.7,0.]])]
   masks=[torch.zeros(0,64,64) if empty else torch.ones(1,64,64)]
   loss=sum(crit(m(torch.randn(1,3,64,64)),targets,masks,[crowd]).values())
   self.assertTrue(torch.isfinite(loss)); m.zero_grad(); loss.backward()
 def test_loader_and_empty_eval(self):
  with tempfile.TemporaryDirectory() as td:
   for i in (1,2): Image.new('RGB',(32,32)).save(Path(td)/f'{i}.png')
   co=COCO(); co.dataset={'images':[{'id':i,'file_name':f'{i}.png','height':32,'width':32} for i in (1,2)],'categories':[{'id':4,'name':'scratch'}],'annotations':[{'id':1,'image_id':1,'category_id':4,'bbox':[4,4,12,12],'area':144,'iscrowd':0,'segmentation':[[4,4,16,4,16,16,4,16]]}]}; co.createIndex()
   ds=COCODetection(td,co,{4:1},input_shape=[64,64]); self.assertEqual(len(ds),2)
   np.testing.assert_array_equal(ds[0][0],ds[0][0]); self.assertEqual(ds[1][1].shape,(0,5)); self.assertEqual(ds[1][2].shape,(0,64,64))
   for kind in ('bbox','segm'): self.assertEqual(score(co,[],kind)[0]['AP50'],0.)
 def test_initialization_consistency(self):
  c=read_config('configs/wsd.json'); c['backbone']='resnet50'
  seed_all(11); base=make_model(c,'baseline').state_dict()
  seed_all(11); full=make_model(c,'ec').state_dict()
  for k,v in base.items(): torch.testing.assert_close(v,full[k],rtol=0,atol=0)
  c['backbone']='resnet101'; model=make_model(c,'ec')
  with tempfile.TemporaryDirectory() as td:
   path=Path(td)/'wrong.pt'; torch.save(base,path)
   with self.assertRaisesRegex(ValueError,'depth'): load_init(model,path)
 def test_ega_prototype_connection(self):
  c=read_config('configs/wsd.json'); c['backbone']='resnet50'
  m=make_model(c,'b',False).eval(); inputs=[]
  handle=m.proto_net.register_forward_pre_hook(lambda module,args:inputs.append(args[0].detach().clone()))
  x=torch.randn(1,3,64,64)
  with torch.no_grad():
   before=m(x); m.proto_attention.gamma.fill_(1); after=m(x)
  handle.remove(); self.assertEqual(len(inputs),2)
  e=inputs[0].square().sum(1,keepdim=True)
  torch.testing.assert_close(inputs[1],inputs[0]+inputs[0]*e/(e.sum((2,3),keepdim=True)+1e-6))
  self.assertFalse(torch.equal(inputs[0],inputs[1]))
  for i in range(3): torch.testing.assert_close(before[i],after[i],rtol=0,atol=0)
  self.assertFalse(torch.equal(before[3],after[3]))
 def test_ega_input_contract(self):
  from utils.EGA import EGA
  self.assertIs(EGA,EnergyAttention)
  with self.assertRaises(ValueError): EGA(eps=0)
  with self.assertRaises(ValueError): EGA()(torch.randn(2,4,4))
  with self.assertRaises(TypeError): EGA()(torch.ones(1,2,4,4,dtype=torch.long))
  for dtype in (torch.float16,torch.bfloat16,torch.float32,torch.float64):
   x=torch.zeros(2,3,8,8,dtype=dtype); y=EGA(initial_scale=1)(x)
   self.assertEqual(y.dtype,dtype); self.assertTrue(torch.isfinite(y).all())
 def test_empty_nms(self):
  c=read_config('configs/wsd.json'); c['backbone']='resnet50'; m=make_model(c,'baseline',False).eval()
  with torch.no_grad(): out=m(torch.zeros(1,3,64,64))
  ans=BBoxUtility().decode_nms(out,torch.tensor(get_anchors([64,64]),dtype=torch.float32),1.,.5,[64,64]); self.assertIsNone(ans[0])
if __name__=='__main__': unittest.main(verbosity=2)
