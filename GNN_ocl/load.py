import pickle
from base_graph import base_graph
i = 0
fn = '/home/cjy/GNN_demo/saved/graph.dat'
with open(fn, 'rb') as f:
    while(1):
        summer = pickle.load(f)
        print(str(summer)+str(i))
        i = i+1





