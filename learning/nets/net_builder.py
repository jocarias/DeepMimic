import learning.nets.fc_2layers_1024units as fc_2layers_1024units
#####################[
from learning.my_code  import MyCNN
#####################]

def build_net(net_name, input_tfs, channel_count=None, reuse=False):
    net = None

    if (net_name == fc_2layers_1024units.NAME):
        net = fc_2layers_1024units.build_net(input_tfs, reuse)
    #####################[
    elif net_name == MyCNN.NAME:
        net = MyCNN(input_tfs, channel_count).network
    #####################]
    else:
        assert False, 'Unsupported net: ' + net_name
    
    return net