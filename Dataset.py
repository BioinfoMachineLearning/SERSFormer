
import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np
from pandas import read_csv
import re
import scipy.signal as signal
DATASET_DIR = "."

label_dict = {'No_pest_present':0,'Thiabenzadole_present':1,'Phosmet_present':2,'Carbophenothion_present':3,'Coumaphos_present':4,'Oxamyl_present':5}
Num_classes = len(label_dict)

train = glob.glob(DATASET_DIR+'/DataInSpinach_AllPesticides/**/*.*', recursive=True)


print("Number of Train pest samples => ", len(train))

class SERSData(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(train)
    
    def __getitem__(self, idx):
        pest_data = train[idx]
        #pest_file_name = os.path.basename(pest_data)
        pest_label = os.path.basename(os.path.dirname(pest_data))
        pest_label1 = torch.as_tensor(label_dict[pest_label])
        pest_concen = float(re.findall(r"[-+]?\d*\.\d+|\d+", pest_data)[0])  if pest_label1!=0 else 0.0
        pest_concen = torch.as_tensor(pest_concen)
        pest_intensity = read_csv(pest_data,header=None, index_col=0).T
        pest_intensity_np = np.array(pest_intensity).squeeze(0)
        pest_intensity_np = np.array([np.log1p(p**2) if p >5 else 0.001 for p in pest_intensity_np ])
        #pest_intensity_tensor = torch.from_numpy(pest_intensity_np).float()
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.min()) / (pest_intensity_tensor.max() - pest_intensity_tensor.min())
        #norm_pest_intensity_tensor_withWave = torch.row_stack((norm_pest_intensity_tensor,torch.as_tensor(pest_intensity.columns))).float()
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.mean()) / pest_intensity_tensor.std() 
        win = signal.windows.hann(389)
        intensity_tensor = (signal.convolve(pest_intensity_np, win, mode='same',method='direct') / sum(win))
        norm_pest_intensity_tensor = torch.from_numpy(intensity_tensor).unsqueeze(0).float()
        
        return [pest_label1,pest_concen, norm_pest_intensity_tensor]

class SERSDataPow(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(train)
    
    def __getitem__(self, idx):
        pest_data = train[idx]
        #pest_file_name = os.path.basename(pest_data)
        pest_label = os.path.basename(os.path.dirname(pest_data))
        pest_label1 = torch.as_tensor(label_dict[pest_label])
        pest_concen = float(re.findall(r"[-+]?\d*\.\d+|\d+", pest_data)[0])  if pest_label1!=0 else 0.0
        pest_concen = torch.as_tensor(pest_concen)
        pest_intensity = read_csv(pest_data,header=None, index_col=0).T
        pest_intensity_np = np.array(pest_intensity)
        pest_intensity_tensor = torch.from_numpy(pest_intensity_np).float()
        pest_intensity_power_sp = pest_intensity_tensor.pow(2).abs()
        norm_pest_intensity_tensor = (pest_intensity_power_sp - pest_intensity_power_sp.min()) / (pest_intensity_power_sp.max() - pest_intensity_power_sp.min())
        
        return [pest_label1,pest_concen, norm_pest_intensity_tensor]


class SERSDatav2(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(train)
    
    def __getitem__(self, idx):
        pest_data = train[idx]
        #pest_file_name = os.path.basename(pest_data)
        pest_label = os.path.basename(os.path.dirname(pest_data))
        pest_label1 = torch.as_tensor(label_dict[pest_label])
        pest_concen = float(re.findall(r"[-+]?\d*\.\d+|\d+", pest_data)[0])  if pest_label1!=0 else 0.0
        pest_concen = torch.as_tensor(pest_concen)
        pest_intensity = read_csv(pest_data,header=None, index_col=0).T
        pest_intensity_np = np.array(pest_intensity).squeeze(0)
        pest_intensity_np = np.array([np.log1p(p**2) if p >5 else 0.001 for p in pest_intensity_np ])
        #pest_intensity_tensor = torch.from_numpy(pest_intensity_np).float()
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.min()) / (pest_intensity_tensor.max() - pest_intensity_tensor.min())
        #norm_pest_intensity_tensor_withWave = torch.row_stack((norm_pest_intensity_tensor,torch.as_tensor(pest_intensity.columns))).float()
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.mean()) / pest_intensity_tensor.std() 
        pest_95 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,95) else 0.0 for x in pest_intensity_np]).float()
        pest_85 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,85) else 0.0 for x in pest_intensity_np]).float()
        pest_75 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,75) else 0.0 for x in pest_intensity_np]).float()
        pest_50 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,50) else 0.0 for x in pest_intensity_np]).float()
        win = signal.windows.hann(389)
        intensity_tensor = (signal.convolve(pest_intensity_np, win, mode='same',method='direct') / sum(win))
        
        norm_pest_intensity_tensor = torch.from_numpy(intensity_tensor).unsqueeze(0).float()
        stacked_tensors = torch.row_stack((pest_95,pest_85,pest_75,pest_50))
        return [pest_label1,pest_concen, norm_pest_intensity_tensor,stacked_tensors]
    
test = glob.glob(DATASET_DIR+'/Testing/*/**/*.*', recursive=True)
class SERSDatav2test(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(test)
    
    def __getitem__(self, idx):
        pest_data = test[idx]
        #pest_file_name = os.path.basename(pest_data)
        pest_label = os.path.basename(os.path.dirname(pest_data))
        pest_label1 = torch.as_tensor(label_dict[pest_label])
        pest_concen = float(re.findall(r"[-+]?\d*\.\d+|\d+", pest_data)[0])  if pest_label1!=0 else 0.0
        pest_concen = torch.as_tensor(pest_concen)
        pest_intensity = read_csv(pest_data,header=None, index_col=0).T
        pest_intensity_np = np.array(pest_intensity).squeeze(0)
        pest_intensity_np = np.array([np.log1p(p**2) if p >5 else 0.001 for p in pest_intensity_np ])
        #pest_intensity_tensor = torch.from_numpy(pest_intensity_np).float()
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.min()) / (pest_intensity_tensor.max() - pest_intensity_tensor.min())
        #norm_pest_intensity_tensor_withWave = torch.row_stack((norm_pest_intensity_tensor,torch.as_tensor(pest_intensity.columns))).float()
        #norm_pest_intensity_tensor = (pest_intensity_tensor - pest_intensity_tensor.mean()) / pest_intensity_tensor.std() 
        pest_95 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,95) else 0.0 for x in pest_intensity_np]).float()
        pest_85 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,85) else 0.0 for x in pest_intensity_np]).float()
        pest_75 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,75) else 0.0 for x in pest_intensity_np]).float()
        pest_50 = torch.as_tensor([np.log1p(x) if x>=np.percentile(pest_intensity_np,50) else 0.0 for x in pest_intensity_np]).float()
        win = signal.windows.hann(389)
        intensity_tensor = (signal.convolve(pest_intensity_np, win, mode='same',method='direct') / sum(win))
        
        norm_pest_intensity_tensor = torch.from_numpy(intensity_tensor).unsqueeze(0).float()
        stacked_tensors = torch.row_stack((pest_95,pest_85,pest_75,pest_50))
        return [pest_label1,pest_concen, norm_pest_intensity_tensor,stacked_tensors]
