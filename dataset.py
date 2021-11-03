# 数据预处理
import os

import torch
import torchvision.utils
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import dataset,dataloader
import config


class CaptchaDataset(dataset.Dataset):
    """
    ## 加载数据，数据格式为
    # train: label.png
    # test: index.png
    """

    def __init__(self, root,multi = False, transformer = None,train = True):
        """
        captcha dataset
        :param root: the paths of dataset, 数据类型为 root/label.png ...
        :param transformer: transformer for image
        :param train: train of not
        """
        super(CaptchaDataset, self).__init__()
        assert root
        self.root = root
        self.train = train
        self.transformer = transformer
        self.labels = None
        self.back = None
        if multi:
            paths = [os.path.join(self.root,path) for path in os.listdir(self.root)]
        else:
            paths = [self.root]
        self._extract_images(paths)

    
    def _extract_images(self,paths):
        self.image_paths = []
        self.labels = []
        for path in paths:
            if not os.path.isdir(path):
                continue
            file_paths = os.listdir(path)
            for file_path in file_paths:
                if file_path.endswith(".png") or file_path.endswith(".jpg") or file_path.endswith("jpeg"):
                    self.image_paths.append(os.path.join(path,file_path))
                    name = file_path.split('.')[0]
                    if name.find('-') != -1:
                        label = name.split('-')[1]
                    elif name.find('_') != -1:
                        label = name.split('_')[1]
                    else:
                        label = name
                    if label.isdigit():
                        self.labels.append(0)
                    else:
                        self.labels.append(1)

        assert len(self.image_paths) == len(self.labels) 
        

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        fail = False
        try:
            img = Image.open(image_path)
            if img.mode == 'RGBA':
                r,g,b,a = img.split()
                img.load() # required for png.split()
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=a) # 3 is the alpha channel
                img  =  background
        except Exception as e:
            img = Image.open('2-mc5m.png')
            img = img.convert("RGB")
            fail = True
           
        if fail:
            label = 1
        else:
            label = self.labels[idx]
        target = torch.LongTensor([label])
        if self.transformer:
            img = self.transformer(img)
        return img, target
 

def resizeNormalize(image,imgH, imgW, train = False):
    """
    resize and normalize image
    """
    if train:
        transformer = transforms.Compose(
        [
         transforms.RandomAffine((0.9,1.1)),
         transforms.RandomRotation(6),
         transforms.Resize((imgH, imgW)),
         transforms.ToTensor(),
         transforms.Normalize(mean=config.mean, std=config.std)
         ]
    )
    else:
        transformer = transforms.Compose(
        [
         transforms.Resize((imgH, imgW)),
         transforms.ToTensor(),
         transforms.Normalize(mean=config.mean, std=config.std)
         ]
    )
    return transformer(image)


class CaptchaCollateFn(object):

    def __init__(self,imgH=32, imgW=100, keep_ratio=False,train = False) -> None:
        super().__init__()
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio = keep_ratio
        self.train = train
    
    def __call__(self, batch):
        images, targets = zip(*batch)
        if self.keep_ratio:
            max_ratio = 0.0
            for image in images:
                w,h = image.size
                max_ratio = max(max_ratio,w/float(h))
            imgW = max(int(max_ratio*self.imgH),self.imgW)
        images = [resizeNormalize(image,self.imgH,self.imgW,self.train) for image in images]
        images = torch.stack(images, 0)
        targets = torch.cat(targets, 0)
        return images, targets
        
    

def train_loader(train_path,multi = False,train_rate = config.train_rate,batch_size = config.batch_size,
                 height = config.height, width = config.width,keep_ratio = True,
                 transformer = None):
    """
    
    :param train_path:  the path of training data
    :param batch_size: 
    :param height resize height
    :param width: resize width
    :return: 
    """""
    # if transformer is None:
    #     transformer = transforms.Compose(
    #         [
    #           #transforms.RandomAffine((0.9,1.1)),
    #           #transforms.RandomRotation(8),
    #           transforms.Resize((height, width)),
    #           transforms.ToTensor(),
    #           transforms.Normalize(mean=config.mean,std= config.std)
    #          ]
    #     )
    train_set = CaptchaDataset(train_path,multi = multi, transformer=transformer)
    train_len = int(len(train_set)*train_rate)
    train_data, val_data = torch.utils.data.random_split(train_set,[train_len,len(train_set)-train_len])
    return dataloader.DataLoader(train_data, batch_size=batch_size, shuffle=True,collate_fn= CaptchaCollateFn(height,width,keep_ratio,True)),\
           dataloader.DataLoader(val_data, batch_size=batch_size, shuffle=True,collate_fn= CaptchaCollateFn(height,width,keep_ratio,False))


def test_loader(test_path,batch_size = config.test_batch_size, height = config.height,
                width = config.width,keep_ratio = True,transformer = None):
    """

    :param test_path:
    :param batch_size:
    :param x: resize
    :param y:
    :return:
    """
    # if transformer is None:
    #     transformer = transforms.Compose(
    #     [transforms.Resize((height, width)),
    #      transforms.ToTensor(),
    #      transforms.Normalize(mean=config.mean, std=config.std)
    #      ]
    # )
    test_set = CaptchaDataset(test_path,train = False, transformer=transformer)
    return dataloader.DataLoader(test_set, batch_size=batch_size, shuffle=False,collate_fn = CaptchaCollateFn(height,width,keep_ratio,False))



if __name__ == '__main__':
     height,width = 32,100
    #  transformer = transforms.Compose(
    #     [
    #         #transforms.RandomAffine((0.9, 1.1)),
    #         #transforms.RandomRotation(8),
    #         transforms.Resize((32, int(width/(height/3)))),
    #         transforms.ToTensor(),
    #     ]
    #  )
     path = '/Users/sjhuang/Documents/docs/dataset/train'
     train_loade,val_loader = train_loader(path,multi = True,transformer = None)
     imgs, targets  = next(iter(train_loade))
     grid_img = torchvision.utils.make_grid(imgs,nrow = 4)
     plt.imshow(grid_img.permute(1, 2, 0))
     plt.imsave(f"pres/preprocessed_{height}_{width}.jpg",grid_img.permute(1, 2, 0).numpy())
     # num = 0
     # for imgs, targets, target_lens  in train_loader:
     #     num += len(imgs)
     #     logger.info(f"imgs:{imgs.shape}, {num}")



