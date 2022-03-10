import torch
import torch.nn.functional as F
import cv2

def Loss(p2, p3, p4, p5, out, y, lamb=0.25):
    l_all = F.cross_entropy(out, y)
    
    N, H, W = y.size()
    y = y.reshape((N, -1, H, W)).float()
    y = F.interpolate(y, size=(64, 64))
    l2 = F.cross_entropy(p2, y.squeeze().long())

    y = F.interpolate(y, size=(32, 32))
    l3 = F.cross_entropy(p3, y.squeeze().long())

    y = F.interpolate(y, size=(16, 16))
    l4 = F.cross_entropy(p4, y.squeeze().long())

    y = F.interpolate(y, size=(8, 8))
    l5 = F.cross_entropy(p5, y.squeeze().long())
    
    l = (l2 + l3 + l4 + l5)/4 + lamb * l_all
    return l, l2, l3, l4, l5, l_all