import argparse,json
from experiment import read_config,check_data
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/wsd.json'); p.add_argument('--annotations-only',action='store_true'); a=p.parse_args()
    print(json.dumps(check_data(read_config(a.config),not a.annotations_only),indent=2))
