import torch
import numpy as np
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.mixture import GaussianMixture

np.random.seed(0)
torch.manual_seed(0)

class Generator(torch.nn.Module):

    def __init__(self, input_size = 10, normalize = True):
        super(Generator, self).__init__()
        self.process_noise  = torch.nn.Sequential(*_dense_layer(input_size,49, normalize = normalize))
        
        layers = [*_conv_block(1,64,normalize),torch.nn.Upsample(scale_factor=2),
                        *_conv_block(64,32,normalize),torch.nn.Upsample(scale_factor=2),
                        *_conv_block(32,16,normalize),*_conv_block(16,1,normalize, activation=torch.nn.Sigmoid())]
        
        self.generate_image = torch.nn.Sequential(*layers)

        return

    def forward(self, noise):

        img = self.process_noise(noise)
        
        img = torch.reshape(img,shape = (-1,1,7,7))
        
        img = self.generate_image(img)
        return img

class ConvGenerator(torch.nn.Module):
    def __init__(self, latent_size, normalize = True):
        super(ConvGenerator, self).__init__()
        self.generator = nn.Sequential(
                    # in: latent_size x 1 x 1

                    nn.ConvTranspose2d(latent_size, 64, kernel_size=5, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    # out: 64 x 4 x 4

                    nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    # out: 32 x 8 x 8

                    nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(True),
                    # out: 16 x 16 x 16

                    nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=0, bias=False, dilation=1),
                    nn.Tanh()
                    # out: 1 x 64 x 64
                )
    def forward(self,X):
        return self.generator(X)



class ConvGeneratorCIFAR(torch.nn.Module):
    def __init__(self, latent_size, normalize = True):
        super(ConvGeneratorCIFAR, self).__init__()
        self.generator = nn.Sequential(
                    # in: latent_size x 1 x 1

                    nn.ConvTranspose2d(latent_size, 64, kernel_size=4, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(True),
                    # out: 64 x 4 x 4

                    nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(32),
                    nn.ReLU(True),
                    # out: 32 x 8 x 8

                    nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2, padding=0, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(True),
                    # out: 16 x 16 x 16

                    nn.ConvTranspose2d(16, 3, kernel_size=2, stride=2, padding=0, bias=False, dilation=1),
                    nn.Tanh()
                    # out: 3 x 64 x 64
                )
    def forward(self,X):
        return self.generator(X)


class DenseGenerator(torch.nn.Module):

    def __init__(self, latent_size = 10, normalize = True):
        super(DenseGenerator, self).__init__()
        
        
        out_sizes = [64,256,512,784]

        layers = []
        layers.append(torch.nn.Sequential(*_dense_layer(latent_size,out_sizes[0], normalize = normalize)))
        for i in range(1,len(out_sizes)):
            layers.append(torch.nn.Sequential(*_dense_layer(out_sizes[i-1],out_sizes[i], normalize = normalize)))
        
        
        self.generate_image = torch.nn.Sequential(*layers)

        return

    def forward(self, noise):

        img = self.generate_image(noise)
        
        img = torch.reshape(img,shape = (-1,1,28,28))
        
       
        return img



class ConvDiscriminator(torch.nn.Module):
    def __init__ (self, normalize = True):
        super(ConvDiscriminator,self).__init__()
        self.discriminator = nn.Sequential(
            # in: 1 x 28 x 28

            nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 4 x 32 x 32

            nn.Conv2d(4, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 8 x 16 x 16

            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 16 x 8 x 8

            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),

            
            # out: 32 x 4 x 4

            nn.Conv2d(32, 1, kernel_size=4, stride=1, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
            nn.Sigmoid())
    def forward(self, X):
        return self.discriminator(X)
    
    def get_latent_rep(self,X):
        for layer in self.discriminator[:-3]:
            X = layer(X)
        return nn.Flatten()(X)





class ConvDiscriminatorCIFAR(torch.nn.Module):
    def __init__ (self, normalize = True):
        super(ConvDiscriminatorCIFAR,self).__init__()
        self.discriminator = nn.Sequential(
            # in: 3 x 32 x 32

            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 64 x 16 x 16

            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 128 x 8 x 8

            nn.Conv2d(16, 32, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            # out: 256 x 4 x 4

            nn.Conv2d(32, 64, kernel_size=2, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),

            
            # out: 32 x 4 x 4

            nn.Conv2d(64, 1, kernel_size=2, stride=2, padding=0, bias=False),
            # out: 1 x 1 x 1

            nn.Flatten(),
            nn.Sigmoid())
    def forward(self, X):
        return self.discriminator(X)
    
    def get_latent_rep(self,X):
        for layer in self.discriminator[:-3]:
            X = layer(X)
        return nn.Flatten()(X)        

class Discriminator(torch.nn.Module):

    def __init__(self, input_size = 10, normalize = True):
        super(Discriminator, self).__init__()
        
        
        layers = [*_conv_block(1,64,normalize),torch.nn.MaxPool2d(kernel_size=(2,2)),
                        *_conv_block(64,32,normalize),torch.nn.MaxPool2d(kernel_size=(2,2)),
                        *_conv_block(32,16,normalize),torch.nn.MaxPool2d(kernel_size=(7,7)),*_conv_block(16,1,normalize, activation=torch.nn.Sigmoid())]
        
        self.discriminate = torch.nn.Sequential(*layers)

        return

    def forward(self, x):

        out = self.discriminate(x)
        out = torch.reshape(out,shape = (-1,1))
        return out
        



def _dense_layer(in_dim,out_dim, normalize ,activation = torch.nn.ReLU(inplace=True)):
    
    layers = []
    layers.append(torch.nn.Linear(in_dim, out_dim))
    if normalize:
            layers.append(torch.nn.BatchNorm1d(out_dim, 0.8))
    layers.append(activation)

    return layers

def _conv_block(in_dim,out_dim, normalize , activation = torch.nn.ReLU(inplace=True)):
    
    layers = []
    
    layers.append(torch.nn.Conv2d(in_dim, out_dim, kernel_size = (3,3), padding = 1))
    if normalize:
        layers.append(torch.nn.BatchNorm2d(out_dim, 0.8))
    layers.append(activation)

    return layers



class BaselineModel:
    def __init__(self, pca_components = 2, nu = 0.5, gmm_components = 5, use_pca = True):
        self.pc = PCA(n_components = pca_components)
        self.svm = OneClassSVM(nu = nu)
        self.gmm = GaussianMixture(gmm_components)
        self.thresh = 0
        self.use_pca = use_pca

    def fit(self,X):
        if self.use_pca:
            data = self.pc.fit_transform(X)
            self.gmm.fit(data)
            self.thresh = np.min(self.gmm.score_samples(data))
        else:
            data = X

        self.svm.fit(data)
        

    def predict(self, X):
        if self.use_pca:
            data = self.pc.transform(X)
        else:
            data = X
        return self.svm.predict(data)

    def predict_gmm(self, X):
        data = self.pc.transform(X)
        scores = self.gmm.score_samples(data)  
        preds = np.array(scores>self.thresh,np.int)
        return preds
