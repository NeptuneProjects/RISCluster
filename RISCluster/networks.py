import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3,5), stride=(2,4), padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # <---- Experimental
            # nn.ReLU(True), # <---- Experimental
            nn.Flatten(),
            nn.Linear(1024, 16),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x

# Decoder Layers
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.latent2dec = nn.Sequential(
            nn.Linear(16, 1024),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, kernel_size=3, stride=(1,2), padding=1), # <---- Experimental
            # nn.ReLU(True),  # <---- Experimental
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, kernel_size=(3,5), stride=(2,4), padding=(1,2)),
        )

    def forward(self, x):
        x = self.latent2dec(x)
        x = x.view(-1, 64, 4, 4)
        x = self.decoder(x)
        return x[:,:,1:,1:]
        # return x

class AEC(nn.Module):
    def __init__(self):
        super(AEC, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        return x, z


# # Encoder Layers
# class Encoder(nn.Module):
#     def __init__(self):
#         super(Encoder, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             # nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # <---- Experimental
#             # nn.ReLU(True), # <---- Experimental
#             nn.Flatten(),
#             nn.Linear(2048, 32),
#             nn.ReLU(True)
#         )
#
#     def forward(self, x):
#         x = self.encoder(x)
#         return x
#
# # Decoder Layers
# class Decoder(nn.Module):
#     def __init__(self):
#         super(Decoder, self).__init__()
#         self.latent2dec = nn.Sequential(
#             nn.Linear(32, 2048),
#             nn.ReLU(True)
#         )
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=1),
#             nn.ReLU(True),
#             nn.ConvTranspose2d(8, 1, kernel_size=3, stride=2, padding=1),
#         )
#
#     def forward(self, x):
#         x = self.latent2dec(x)
#         x = x.view(-1, 64, 4, 8)
#         x = self.decoder(x)
#         return x[:,:,1:,1:]
#         # return x

def init_weights(m):
    if type(m) in [nn.Linear, nn.Conv2d, nn.ConvTranspose2d]:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

# Clustering Layer
class ClusteringLayer(nn.Module):
    def __init__(self, n_clusters, n_features=16, alpha=1.0, weights=None):
        super(ClusteringLayer, self).__init__()
        self.n_features = n_features
        self.n_clusters = n_clusters
        self.alpha = alpha
        if weights is None:
            initial_weights = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_weights)
        else:
            initial_weights = weights
        self.weights = nn.Parameter(initial_weights)

    def forward(self, x):
        x = x.unsqueeze(1) - self.weights
        x = torch.mul(x, x)
        x = torch.sum(x, dim=2)
        x = 1.0 + (x / self.alpha)
        x = 1.0 / x
        x = x ** ((self.alpha +1.0) / 2.0)
        x = torch.t(x) / torch.sum(x, dim=1)
        x = torch.t(x)
        return x

# class AEC(nn.Module):
#     def __init__(self):
#         super(AEC, self).__init__()
#         self.encoder = Encoder()
#         self.decoder = Decoder()
#
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

class DCM(nn.Module):
    def __init__(self, n_clusters):
        super(DCM, self).__init__()
        self.n_clusters = n_clusters
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.clustering = ClusteringLayer(self.n_clusters, n_features=16, alpha=1.0)

    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)
        q = self.clustering(z)
        return q, x, z
