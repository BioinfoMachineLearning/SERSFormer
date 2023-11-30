
import os
import glob
import torch
import numpy as np
from pandas import read_csv, DataFrame
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import re
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.signal as signal

DATASET_DIR = "/home/aghktb/AI-SERS"
label_dict = {'No_pest_present':0,'Thiabenzadole_present':1,'Phosmet_present':2,'Carbophenothion_present':3,'Coumaphos_present':4,'Oxamyl_present':5}
Num_classes = len(label_dict)

train = glob.glob(DATASET_DIR+'/DataInSpinach_AllPesticides/**/*.*', recursive=True)

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
        #pest_concen = torch.as_tensor(pest_concen)
        pest_intensity = read_csv(pest_data,header=None, index_col=0)
        pest_intensity_np = np.array(pest_intensity).squeeze(1)
        pest_95 = [np.log1p(x) if x>=np.percentile(pest_intensity_np,95) else 0.0 for x in pest_intensity_np]
        pest_85 = [np.log1p(x) if x>=np.percentile(pest_intensity_np,85) else 0.0 for x in pest_intensity_np]
        pest_75 = [np.log1p(x) if x>=np.percentile(pest_intensity_np,75) else 0.0 for x in pest_intensity_np]
        pest_50 = [np.log1p(x) if x>=np.percentile(pest_intensity_np,50) else 0.0 for x in pest_intensity_np]
        pest_intensity_np = np.array([np.log1p(p**2) if p >1 else 0.001 for p in pest_intensity_np ])
        
        
        #pest_intensity_tensor = torch.from_numpy(pest_intensity_np).float()
        norm_pest_intensity_tensor = (pest_intensity - pest_intensity.min()) / (pest_intensity.max() - pest_intensity.min())
        norm_pest_intensity_tensor['conc'] = [pest_concen for i in range(len(pest_intensity))]
        norm_pest_intensity_tensor['orig_in'] = pest_intensity
        norm_pest_intensity_tensor['zscore'] = (pest_intensity - pest_intensity.mean()) / pest_intensity.std()
        win = signal.windows.hann(389)
        filtered = signal.convolve(pest_intensity_np, win, mode='same',method='direct') / sum(win)
        #filtered = (filtered - filtered.mean()) / filtered.std()
        norm_pest_intensity_tensor['convolve'] = filtered
        norm_pest_intensity_tensor['label'] = [pest_label1 for i in range(len(pest_intensity))]
        norm_pest_intensity_tensor['95perc'] = pest_95
        norm_pest_intensity_tensor['85perc'] = pest_85
        norm_pest_intensity_tensor['75perc'] = pest_75
        norm_pest_intensity_tensor['50perc'] = pest_50
        return norm_pest_intensity_tensor
    

Data1 = SERSData(DATASET_DIR)
df1 = [Data1[i] for i in range(len(Data1))]
df = df1[0]
from scipy.signal import find_peaks
corr = torch.corrcoef(df['orig_in'],df['conc'])


#label1 = []
#for df in df1:
#
#    #label1.append(df['label'][0])
#    if(df['label'].iloc[0]==2):
#        
#        plt.plot(df['convolve'][df['conc']==5.],color='yellow')
#        plt.plot(df['convolve'][df['conc']==10.],color='grey')
#        plt.plot(df['convolve'][df['conc']==2.],color='orange')
#        plt.plot(df['convolve'][df['conc']==1],color='red')
#       
#        plt.plot(df['convolve'][df['conc']==0.],color='green')
#        plt.plot(df['convolve'][df['conc']==0.5],color='blue')