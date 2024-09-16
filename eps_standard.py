import torch
import numpy as np


cf10_pt = "/data/gpfs/projects/punim2205/STRAP/strap/loaded_data/cifar10/sd121/5000_cifar10_eps.pt"
cf10_eps_data = torch.load(cf10_pt)

cf10_nat_eps_list = []
for score in cf10_eps_data:
    cf10_nat_eps_list.append(torch.norm(score.flatten(), p=2).cpu().item())

cf10_eps_standard = cf10_nat_eps_list

imgnet_pt = "/data/gpfs/projects/punim2205/STRAP/strap/loaded_data/imagenet/sd121/5000_imagenet_eps.pt"
imgnet_eps_data = torch.load(imgnet_pt) #保存方式与cf10不同

imgnet_nat_eps_list = []
for score in imgnet_eps_data:
    imgnet_nat_eps_list.append(torch.norm(score.flatten(), p=2).cpu().item())

imagenet_eps_standard = imgnet_nat_eps_list