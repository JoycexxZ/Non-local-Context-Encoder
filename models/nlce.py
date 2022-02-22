import torch.nn as nn
import torch.nn.functional as F
import torch

from encoding import Encoding

class NLCE(nn.Module):
    def __init__(self, C_in, C1, D=128, K=32):
        super(NLCE, self).__init__()
        self.D, self.K, self.C1, self.C_in= D, K, C1, C_in
        self.encoder = Encoding(self.D, self.K)
        self.conv1_theta = nn.Conv2d(C_in, self.C1, 1)
        self.conv1_phi = nn.Conv2d(C_in, self.C1, 1)
        self.conv1_g = nn.Conv2d(C_in, self.C1, 1)
        self.conv2 = nn.Conv2d(self.C1, C_in, 1)
        self.conv3 = nn.Conv2d(C_in, self.D, 1)
        self.fc = nn.Linear(D, C_in)


    def forward(self, X):

        B, C_in, H, W = X.shape

        theta = self.conv1_theta(X)
        phi = self.conv1_phi(X)
        g = self.conv1_g(X)

        theta = theta.reshape((B, self.C, -1))
        phi = phi.reshape((B, self.C, -1)).tranpose(1, 2)
        g = g.reshape((B, self.C, -1))

        f = F.softmax(torch.bmm(theta, phi))
        y = torch.bmm(f, g)

        y = y.reshape(B, self.C, H, W)
        z = self.conv2(y) + X

        z_ = self.conv3(z)

        
        e = self.encoder(z_)
        e = F.batch_norm(e)
        e = F.relu(e)
        E = e.sum(dim=1)

        gamma = self.fc(E)
        gamma = F.sigmoid(gamma)

        return torch.mul(z, gamma)

        





