"""Read a trusted checkpoint and save chosen validation images without training."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from common import load_checkpoint,choose_device,save_rgb
from data import MSRSDataset
from models import Generator
from losses import content_loss

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--checkpoint',required=True)
 p.add_argument('--data-root',required=True)
 p.add_argument('--output',required=True)
 p.add_argument('--device',default='cpu')
 p.add_argument('--names',nargs='+',default=['00007N','01549D'])
 args=p.parse_args();torch.set_num_threads(4)
 out=Path(args.output)
 if out.exists() and any(out.iterdir()):raise FileExistsError('Choose a new output folder')
 ck=load_checkpoint(args.checkpoint);cfg=ck['config'];device=choose_device(args.device)
 if not set(args.names)<=set(ck['manifest']['val']):raise ValueError('Use validation sample names from split.json')
 ds=MSRSDataset(args.data_root,cfg['data']['train_split'],names=args.names,ir_dir=cfg['data']['ir_dir'],vis_dir=cfg['data']['vis_dir'])
 g=Generator(**cfg['model']).to(device).eval();g.load_state_dict(ck['generator'],strict=True)
 out.mkdir(parents=True,exist_ok=True);report=[]
 with torch.no_grad():
  for index in range(len(ds)):
   item=ds[index]
   ir,vis=item['ir'][None].to(device),item['vis'][None].to(device)
   pred=g(ir,vis);loss,_=content_loss(pred,ir,vis,cfg['loss'])
   save_rgb(pred[0],out/(item['name']+'.png'))
   values={'name':item['name'],'step':ck['step'],'min':pred.min().item(),'max':pred.max().item(),'std':pred.std().item(),'content':loss.item(),'black_fraction':(pred<=-0.99999).float().mean().item(),'white_fraction':(pred>=0.99999).float().mean().item()}
   report.append(values);print(json.dumps(values),flush=True)
 (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

if __name__=='__main__':main()
