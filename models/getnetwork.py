import sys
from models import *
import torch.nn as nn

def get_network(network, in_channels, num_classes, **kwargs):

    if network == 'GobletNet':
        net = GobletNet(in_channels, num_classes)
    elif network == 'deeplabv3+':
        net = deeplabv3plus_resnet50(in_channels, num_classes, True)
    elif network == 'unet':
        net = unet(in_channels, num_classes)
    elif network == 'unet_plusplus' or network == 'unet++':
        net = unet_plusplus(in_channels, num_classes)
    elif network == 'resunet':
        net = res_unet(in_channels, num_classes)
    elif network == 'u2net':
        net = u2net(in_channels, num_classes)
    elif network == 'u2net_s':
        net = u2net_small(in_channels, num_classes)
    elif network == 'unet3+':
        net = unet_3plus(in_channels, num_classes)
    elif network == 'unet3+_ds':
        net = unet_3plus_ds(in_channels, num_classes)
    elif network == 'swinunet':
        net = swinunet(num_classes, 224)  # img_size = 224
    elif network == 'wavesnet':
        net = wsegnet_vgg16_bn(in_channels, num_classes)
    elif network == 'mwcnn':
        net = mwcnn(in_channels, num_classes)
    elif network == 'alnet':
        net = Aerial_LaneNet(in_channels, num_classes)
    elif network == 'wds':
        net = WDS(in_channels, num_classes)
    elif network == 'dcr':
        net = DCR(in_channels, num_classes)
    elif network == 'fusionnet':
        net = FusionNet(in_channels, num_classes)
    elif network == 'xnet':
        net = XNet(in_channels, num_classes)
    elif network == 'xnet_v2':
        net = XNet_v2(in_channels, num_classes)
    else:
        print('the network you have entered is not supported yet')
        sys.exit()
    return net
