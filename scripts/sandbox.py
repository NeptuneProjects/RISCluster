import sys
sys.path.insert(0, '../RISCluster/')

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import importlib as imp

# import networks
# imp.reload(networks)
# from  networks import AEC, DCEC
#
# import plotting
# imp.reload(plotting)
# from plotting import view_DCEC_output as w_spec
#
# import processing
# imp.reload(processing)
#
# import production
# imp.reload(production)
#
# import utils
# imp.reload(utils)

# Encoder Layers
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

class Flattener(nn.Module):
    def __init__(self):
        super(Flattener, self).__init__()
        self.flattener = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 32),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.flattener(x)
        return x

# Decoder Layers
class Unflattener(nn.Module):
    def __init__(self):
        super(Unflattener, self).__init__()
        self.latent2dec = nn.Sequential(
            nn.Linear(32, 2048),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.latent2dec(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=5, stride=2, padding=0),
        )

    def forward(self, x):
        x = x.view(-1, 64, 4, 8)
        x = self.decoder(x)
        return x




x = torch.rand(800, 1, 64, 128)
encoder = Encoder()
flattener = Flattener()
unflattener = Unflattener()
decoder = Decoder()


z_ = encoder(x)
z = flattener(z_)
x_r_ = unflattener(z)
x_r = decoder(x_r_)

print('-' * 60)
print(f'INPUT: {x.size()}')
print(f'CONV2D: {z_.size()}')
print(f'FLATTEN: {z.size()}')
print(f'UNFLATTEN: {x_r_.size()}')
print(f'CONV2DT: {x_r.size()}')






#
