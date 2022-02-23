import torch
import torch.nn.functional as F

class loss():
    def loss(self, p2, p3, p4, p5, out, y, lamb=0.25):
        l2 = self.segmentation_loss(p2, y)
        l3 = self.segmentation_loss(p3, y)
        l4 = self.segmentation_loss(p4, y)
        l5 = self.segmentation_loss(p5, y)
        l = self.segmentation_loss(out, y)

        l = (l2 + l3 + l4 + l5)/4 + lamb * l
        return l

    def segmentation_loss(self, x, y):
        return F.cross_entropy(x, y)