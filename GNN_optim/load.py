import pickle
from base_graph import base_graph
i = 0
fn = '/home/chen/Documents/denseReg-master/exp/train_cache/nyu_training_s2_f128_daug_um_v1/graph.dat'
with open(fn, 'rb') as f:
    while(1):
        summer = pickle.load(f)
        print(str(summer)+str(i))
        i = i+1





