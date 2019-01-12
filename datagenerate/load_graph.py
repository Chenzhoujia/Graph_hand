import pickle
# getdata chen_begin
class base_graph():
    globals = None
    nodes_pt = None
    nodes_gt = None
    edges = None
    receivers = None
    senders = None
    def __init__(self, nodes_pt_,nodes_gt_):
        self.nodes_pt = nodes_pt_
        self.nodes_gt = nodes_gt_
# getdata chen_end
i = 0
fn = '/home/chen/Documents/denseReg-master/exp/train_cache/nyu_training_s2_f128_daug_um_v1/graph.dat'
with open(fn, 'rb') as f:
    while(1):
        summer = pickle.load(f)
        print(str(summer)+str(i))
        i = i+1




