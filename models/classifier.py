import torch
import torch.nn as  nn
import torch.nn.functional as F
from models.resnet import ResNet
from tqdm import tqdm
import os
import time
import wandb

class Classifier(object):
    """
    """
    def __init__(self, input_shape, out_channel,save_path,early_stop = 100, predict = False):
        super().__init__(), 
        self.val_epochs = []
        self.train_epochs = []
        self.acc = []
        self.val_loss = []
        self.f1_score = []
        self.train_loss = []
        self.input_shape = input_shape
        self.out_channel = out_channel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = ResNet(self.input_shape,self.out_channel)
        self.net.to(self.device)
        self.best_score = 0.0
        self.early_stop = early_stop
        self.early_stop_count = 0
        self.save_path = save_path
        self.predict = predict 
        if not self.predict:
            self.experiment = wandb.init('image classifer')
    
    def parameters(self):
        return self.net.parameters()

    def cal_acc(self, out, labels):
        out = out.argmax(dim=1)
        labels= labels.reshape((-1))
        c = (out == labels)
        return torch.sum(c).item()
    
    def eval(self, Loss,epoch, data_loader):
        self.net.eval()
        self.val_epochs.append(epoch)
        total_loss = 0
        total_count = 0.0
        total_acc = 0
        pbar = tqdm(total=len(data_loader), desc=f"Eval, epoch:{epoch}")
        with torch.no_grad():
             for imgs, labels in data_loader:
                 imgs, labels = imgs.to(self.device), labels.to(self.device)
                 out = self.net(imgs)
                 loss = Loss(out,labels)
                 total_loss += loss.item()
                 total_count += imgs.shape[0]
                 total_acc += self.cal_acc(out,labels)
                 pbar.update(1)
        pbar.close()
        self.val_loss.append(total_loss/total_count)
        self.acc.append(total_acc/total_count)
        self.experiment.log({
        'val loss': self.val_loss[-1],
        'val acc':self.acc[-1],
        'epoch': epoch,
        'images': wandb.Image(imgs[-1].cpu(),caption=f'Real:{labels[-1].item()}, Pred:{out[-1].argmax().item()}'),
    })
        if self.best_score < self.acc[-1]:
            self.best_score = self.acc[-1]
            self.early_stop_count = 0
            self.save()
        else:
            self.early_stop_count += 1            

    def train(self, Optim,Loss, epoch, data_loader):
        self.net.train()
        self.train_epochs.append(epoch)
        total_loss = 0.0
        total_count = 0
        pbar = tqdm(total=len(data_loader), desc=f"Train, epoch:{epoch}")
        for imgs, labels in data_loader:
            Optim.zero_grad()
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            out = self.net(imgs)
            loss = Loss(out,labels)
            loss.backward()
            Optim.step()
            total_loss += loss.item()
            total_count += imgs.shape[0]
            pbar.update(1)
        pbar.close()
        self.train_loss.append(total_loss/total_count)
        self.experiment.log({
        'train loss':self.train_loss[-1],
        'epoch': epoch
        })
    
    def save(self):
        pid = os.getpid()
        day = time.strftime('%Y%m%d', time.localtime(time.time()))
        model_name = f'{day}_{pid}_model.pt'
        path = os.path.join(self.save_path,model_name)
        torch.save(self.net.state_dict(),path)
    
    
    def desc(self):
        print(f'Epoch {self.train_epochs[-1]}, Train loss {self.train_loss[-1]}, Val loss {self.val_loss[-1]} \
            Val acc {self.acc[-1]}')
    
    def runing(self):
        return self.early_stop_count < self.early_stop
    
    def load(self,model_path):
        self.net.load_state_dict(model_path)
    
    def predict(self,images):
        self.net.eval()
        out = self.net(images)
        preds = out.argmax(dim=1)
        return preds
                
