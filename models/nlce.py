import torch.nn as nn
import torch.nn.functional as F
import torch


class NLCE(nn.Module):
    def __init__(self, C, D=128, K=32):
        super(NLCE, self).__init__()

        self.D, self.K, self.C = D, K, C
        self.codewords = nn.Parameter(torch.Tensor(K, D), requires_grad=True)
        self.scale = nn.Parameter(torch.Tensor(K), requires_grad=True)
        self.reset_params()

    def reset_params(self):
        std = 1. / ((self.K * self.D) ** (1 / 2))
        self.codewords.data.uniform_(-std, std)
        self.scale.data.uniform_(-1, 0)

    def forward(self, X):

        B, C_in, H, W = X.shape

        conv1 = nn.Conv2d(C_in, self.C, 1)
        conv2 = nn.Conv2d(self.C, C_in, 1)
        conv3 = nn.Conv2d(C_in, self.D, 1)

        theta = conv1(X)
        fai = conv1(X)
        g = conv1(X)

        theta = theta.reshape((B, self.C, -1))
        fai = fai.reshape((B, self.C, -1)).tranpose(1, 2)
        g = g.reshape((B, self.C, -1))

        f = F.softmax(torch.bmm(theta, fai))
        y = torch.bmm(f, g)

        y = y.reshape(B, self.C, H, W)
        z = conv2(y) + X

        z_ = conv3(z)




