import torch
import torch.nn.functional as F

def Loss(p2, p3, p4, p5, out, y, lamb=0.25):
    l2 = segmentation_loss(p2, y)
    l3 = segmentation_loss(p3, y)
    l4 = segmentation_loss(p4, y)
    l5 = segmentation_loss(p5, y)
    l_all = segmentation_loss(out, y)
    # print(p2.size(), p3.size(), p4.size(), p5.size(), out.size(), y.size())
    l = (l2 + l3 + l4 + l5)/4 + lamb * l_all
    return l, l2, l3, l4, l5, l_all

def segmentation_loss(x, y):
    return F.cross_entropy(x, y)