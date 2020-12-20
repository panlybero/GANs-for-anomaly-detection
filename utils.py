import torch
from sklearn.metrics import precision_score,recall_score,f1_score
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from utils import *
from model import *
import tqdm
from torchvision.utils import save_image
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import os
import queue
import json


np.random.seed(0)
torch.manual_seed(0)


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, labels, transform, masks):
        'Initialization'
        self.labels = labels
        self.X = X
        self.transforms = transform
        

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        x = self.X[index]

        # Load data and get label
        
        y = self.labels[index]
        
        if self.transforms:
              x = self.transforms(x)

        return x, y



def train_discriminator(disc, gen, batch_size,device,latent_size,real_images, disc_opt, replay = True, old_fakes = None):
    disc_opt.zero_grad()

    real_preds = disc(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device) *0.9
    real_loss = torch.nn.functional.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = gen(latent)
    
    fake_labels = torch.zeros(fake_images.size(0), 1, device=device) +0.1
    fake_preds = disc(fake_images)
    fake_loss = torch.nn.functional.binary_cross_entropy(fake_preds, fake_labels)
    fake_score = torch.mean(fake_preds).item()
    
    old_fake_loss = 0
    if replay:
        for i in range(1):
            if not old_fakes.empty():

                old_fake_imgs = old_fakes.get()
                old_fake_targets = torch.zeros(old_fake_imgs.size(0), 1, device=device) +0.1

                old_fake_preds = disc(old_fake_imgs.cuda())
                old_fake_loss += torch.nn.functional.binary_cross_entropy(old_fake_preds, old_fake_targets)

                if np.random.randn()<0.1 and not old_fakes.full():
                    old_fakes.put(old_fake_imgs.cpu().detach())

        if not old_fakes.full():
            
            old_fakes.put(fake_images.cpu().detach())

    else:
        old_fake_loss = fake_loss
    

    

    total_loss = real_loss + (fake_loss + old_fake_loss) *0.5
    total_loss.backward()
    disc_opt.step()
    return total_loss.item(), real_score, fake_score


def train_generator(gen, disc, batch_size, gen_opt, latent_size, device):
    
    gen_opt.zero_grad()
    
    
    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = gen(latent)
    
    
    preds = disc(fake_images)
    labels = torch.ones(batch_size, 1, device=device) *0.9
    loss = torch.nn.functional.binary_cross_entropy(preds, labels)
    
    
    loss.backward()
    gen_opt.step()
    
    return loss.item()

def save_samples(gen, latent, num = 0, show=True):

    sample_dir = 'generated'
    os.makedirs(sample_dir, exist_ok=True)

    fake_images = gen(latent[:1])
    fake_fname = 'generated-images-{0:0=4d}.png'.format(num)
    save_image(fake_images.detach(), os.path.join(sample_dir, fake_fname), nrow=8)
    print('Saving', fake_fname)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


def get_test_latent(disc,loader ):
    torch.cuda.empty_cache()
    y = []
    labels = []
    for x in loader:
        latent = disc.get_latent_rep(x[0].cuda()).detach().cpu()
        y.append(latent)
        labels.append(x[1])

    return y, labels

def get_test_disc_pred(disc,loader, return_x = False):
    torch.cuda.empty_cache()
    y = []
    labels = []
    X = []
    for x in loader:
        
        if return_x:
            X.append(x[0].detach().cpu())
        latent = disc(x[0].cuda()).detach().cpu()
        y.append(latent)
        labels.append(x[1])

    if return_x:
        
        return y,labels,X
    return y, labels



def evaluate_baseline(model,loader, target= 0, vsize = 28*28):
    y = []
    labels = []
    for x in loader:
    
        pred = model.predict((x[0]).detach().cpu().numpy().reshape(-1,vsize))
        y.append(pred)
        labels.append(x[1])

    y = (np.concatenate(y)+1)*0.5
    labels = torch.cat(labels).numpy()
    
    labels = np.array(labels == target,np.int)


    return y, labels, precision_score(labels,y), recall_score(labels,y), f1_score(labels,y)

def evaluate_baseline_gmm(model,loader, target= 0, vsize = 28*28):
    y = []
    labels = []
    for x in loader:
    
        pred = model.predict_gmm((x[0]).detach().cpu().numpy().reshape(-1,vsize))
        
        y.append(pred)
        labels.append(x[1])

    y = np.concatenate(y)
    labels = torch.cat(labels).numpy()
    
    labels = np.array(labels == target,np.int)


    return y, labels, precision_score(labels,y), recall_score(labels,y), f1_score(labels,y)


def evaluate_latent_ocsvm(model,loader, target = 0):
  
  
    latent,labels = get_test_latent(loader=loader)
    latent = torch.cat(latent).numpy()
    y = model.predict(latent)

    y = (y+1)*0.5
    labels = torch.cat(labels).numpy()
    
    labels = np.array(labels == target,np.int)


    return y, labels, precision_score(labels,y), recall_score(labels,y), f1_score(labels,y)



def evaluate_model(disc,loader, return_x = False,target =0):
    
    if not return_x:
        y,labels = get_test_disc_pred(disc,loader)
    else:
        y,labels = get_test_disc_pred(disc,loader)

    labels = torch.cat(labels)
    y = torch.cat(y)

    labels = labels.numpy()
    labels = np.array(labels == target, np.int)
    y = np.round(y.numpy())

    return y, labels, precision_score(labels,y), recall_score(labels,y), f1_score(labels,y)




def fit(disc, gen, train_loader,epochs, fixed_latent, device,lr_g = 1e-4, lr_d = 1e-4, start_idx=1, replay = True, run_on_test = False, test_loader = None,batch_size = 32, model_name = "novelty_gan"):
    old_fakes = None
    if replay:
          old_fakes = old_fakes = queue.Queue(maxsize = 100)
    model_name = model_name+f"_replay_{replay}_best"
    f1s = []
    torch.cuda.empty_cache()
    disc.cuda()
    gen.cuda()
    disc.train()
    gen.train()
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    best_val_f1 = -np.inf
    # Create optimizers
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr_d, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(gen.parameters(), lr=lr_g, betas=(0.5, 0.999))
    latent_size = fixed_latent.shape[1]
    for epoch in range(epochs):
        loss_g = 0
        loss_d = 0
        counter = 0
        real_score = 0
        fake_score = 0
        for real_images, _ in tqdm.tqdm(train_loader):
            counter+=1
            
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(disc,gen,batch_size,device,latent_size,real_images.cuda(), opt_d, replay = replay, old_fakes=old_fakes)
            # Train generator
            if counter % 1 == 0:#for j in range(5):
                  
                loss_g = train_generator(gen,disc,batch_size,opt_g, latent_size, device)
            #loss_g = 0
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images
        save_samples(gen,fixed_latent,num = epoch+start_idx, show=False)
        if run_on_test:
            _, __, precision, recall,f1 = evaluate_model(disc,test_loader)
            if f1>best_val_f1:
                  best_val_f1 = f1
                  torch.save(gen.state_dict(),os.path.join("./models",model_name+"_generator.model"))
                  torch.save(disc.state_dict(),os.path.join("./models",model_name+"_discriminator.model"))
                  file = open(os.path.join("./models",model_name+"_stats.json"),'w')
                  json.dump({"precision":precision,"recall":recall,"f1":f1},file)
                  file.close()
            f1s.append(f1)
    
    return losses_g, losses_d, real_scores, fake_scores, f1s