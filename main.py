import argparse
from storage import Storage
from train import Train

parser = argparse.ArgumentParser(description="Master's project 10553520 Zizhuo Wang")
parser.add_argument('phase', type=str, help="storage/train/test")
parser.add_argument('-config',type=str,help="path to configuration file")
args = parser.parse_args()

with open(args.config,'r') as f:
    config_dict = dict()
    for line in f.readlines():
        line = line.replace(" ","").strip()
        if line.count(":") > 0:
            head,_,tail = line.partition(":")
            config_dict[head] = tail

if args.phase == 'storage':
    Storage(config_dict).to_mongo()
if args.phase == 'train':
    Train(config_dict).train()