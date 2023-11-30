"""
author: Akshata 
timestamp: Sat July 28 2023 08:10 PM
"""

import os
import glob
from typing import List, Union
from lightning.pytorch.utilities.types import EPOCH_OUTPUT
import torch
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import numpy as np
from pandas import read_csv
from torch import  Tensor
import torch.nn.functional as F
import re
import seaborn as sns
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import wandb
from torchmetrics import MetricCollection
from torchmetrics.classification import BinaryAccuracy, BinaryRecall, BinaryPrecision, BinaryConfusionMatrix, BinaryF1Score,MulticlassAccuracy,MulticlassPrecision,MulticlassRecall,MulticlassF1Score,MulticlassConfusionMatrix
from torchmetrics.regression import MeanSquaredError,R2Score,MeanAbsoluteError

from Model_v2 import TransformerClassifyRegress_con
from argparse import ArgumentParser
import scipy.signal as signal
from Dataset import SERSDatav2,SERSDatav2test
AVAIL_GPUS = [0,1,2]
NUM_NODES = 1
BATCH_SIZE = 32
DATALOADERS = 1
ACCELERATOR = 'gpu'
EPOCHS = 50
ATT_HEAD = 1
ENCODE_LAYERS = 4
DATASET_DIR = "."

label_dict = {'No_pest_present':0,'Thiabenzadole_present':1,'Phosmet_present':2,'Carbophenothion_present':3,'Coumaphos_present':4,'Oxamyl_present':5}
Num_classes = len(label_dict)
"""

torch.set_default_tensor_type(torch.FloatTensor)  # Ensure that the default tensor type is FloatTensor
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Choose the device you want to use

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner to find the best algorithm to use for hardware
    torch.set_default_tensor_type(torch.cuda.FloatTensor)  # Set the default tensor type to CUDA FloatTensor
    torch.set_float32_matmul_precision('medium')  # Set Tensor Core precision to medium

"""

CHECKPOINT_PATH = f"{DATASET_DIR}/Training2/tempo"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)


class SERSClassifyRegress(pl.LightningModule):
    def __init__(self, learning_rate=1e-4,attn_head=ATT_HEAD,encoder_layers=ENCODE_LAYERS,n_class=1, **model_kwargs):
        super().__init__()
        self.test_datasets = ["dataloader0","dataloader1"]
        self.save_hyperparameters()
        self.model = TransformerClassifyRegress_con(attn_head=attn_head,encoder_layers=encoder_layers,n_class=n_class,**model_kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_reg = nn.MSELoss()
        self.metrics_class = MetricCollection([MulticlassAccuracy(num_classes=n_class),
                                         MulticlassPrecision(num_classes=n_class),
                                         MulticlassRecall(num_classes=n_class),
                                         MulticlassF1Score(num_classes=n_class)])
        self.metrics_regress = MetricCollection([ MeanSquaredError(),
                                                 R2Score(),
                                                 MeanAbsoluteError()])
        self.train_metrics_class = self.metrics_class.clone(prefix="train_")
        self.train_metrics_regress = self.metrics_regress.clone(prefix="train_")
        self.valid_metrics_class = self.metrics_class.clone(prefix="valid_")
        self.valid_metrics_regress = self.metrics_regress.clone(prefix="valid_")
        self.test_metrics_class = self.metrics_class.clone(prefix="test_")
        self.test_metrics_regress = self.metrics_regress.clone(prefix="test_")

    def forward(self, pest_sample):
        x = self.model(pest_sample)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=20, eps=1e-10, verbose=True)
        metric_to_track = 'valid_loss'
        return{'optimizer':optimizer,
               'lr_scheduler':lr_scheduler,
               'monitor':metric_to_track}
    
    def training_step(self,batch,batch_idx):
        batch_data = batch[2:]
        
        y_hat = self.forward(batch_data)
        batch_label_class = batch[0].type('torch.LongTensor')
        
        batch_label_class = batch_label_class[:,None].cuda()
        batch_conc = batch[1].to(y_hat[1].dtype)
        batch_conc = batch_conc[:,None]
        class_pred = y_hat[0]
        
        conc_pred = y_hat[1]
       
        loss_class = self.loss_fn(class_pred,batch_label_class.squeeze())
        metric_log_class = self.train_metrics_class(class_pred, batch_label_class.int().squeeze())
        self.log_dict(metric_log_class)
        loss_reg = self.loss_fn_reg(conc_pred,batch_conc)
        metric_log_reg = self.train_metrics_regress(conc_pred, batch_conc.float())
        self.log_dict(metric_log_reg)
        self.log('train_loss_class', loss_class, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_loss_reg', loss_reg, on_step=True, on_epoch=True, sync_dist=True)
        loss = (loss_class+loss_reg)/2
        
        self.log('train_loss',loss, on_step=True, on_epoch=True, sync_dist=True)

        return loss
    
    def validation_step(self,batch,batch_idx):
        batch_data = batch[2:]
        
        y_hat = self.forward(batch_data)
        batch_label_class = batch[0].type('torch.LongTensor')
        batch_label_class = batch_label_class[:,None].cuda()
        batch_conc = batch[1].to(y_hat[1].dtype)
        batch_conc = batch_conc[:,None]
        class_pred = y_hat[0]
        conc_pred = y_hat[1]
        loss_class = self.loss_fn(class_pred,batch_label_class.squeeze())
        metric_log_class = self.valid_metrics_class(class_pred, batch_label_class.int().squeeze())
        self.log_dict(metric_log_class)
        loss_reg = self.loss_fn_reg(conc_pred,batch_conc)
        metric_log_reg = self.valid_metrics_regress(conc_pred, batch_conc.float())
        self.log_dict(metric_log_reg)
        self.log('valid_loss_class', loss_class, on_step=True, on_epoch=True, sync_dist=True)
        self.log('valid_loss_reg', loss_reg, on_step=True, on_epoch=True, sync_dist=True)
        loss = (loss_class+loss_reg)/2
        self.log('valid_loss',loss, on_step=True, on_epoch=True, sync_dist=True)
       
    
    def test_step(self,batch, batch_idx,dataloader_idx):
        batch_data = batch[2:]
        y_hat = self.forward(batch_data)
        batch_label_class = batch[0].type('torch.LongTensor').cuda()
        batch_label_class = batch_label_class[:,None]
        batch_conc = batch[1].to(y_hat[1].dtype)
        batch_conc = batch_conc[:,None]
        class_pred = y_hat[0]
        conc_pred = y_hat[1]
        loss_class = self.loss_fn(class_pred,batch_label_class.squeeze())
        metric_log_class = self.test_metrics_class(class_pred, batch_label_class.int().squeeze())
        self.log_dict(metric_log_class)
        loss_reg = self.loss_fn_reg(conc_pred,batch_conc)
        metric_log_reg = self.test_metrics_regress(conc_pred, batch_conc.float())
        self.log_dict(metric_log_reg)
        

        
        print("Test Data Confusion Matrix: \n")
       
        
        self.log('test_loss_class', loss_class, on_step=True, on_epoch=True, sync_dist=True)
        self.log('test_loss_reg', loss_reg, on_step=True, on_epoch=True, sync_dist=True)
        loss = (loss_class+loss_reg)/2
        
        self.log('test_loss',loss, on_epoch=True, sync_dist=True)
       
        
        return {f'preds_class{dataloader_idx}' : class_pred, f'targets_class{dataloader_idx}' : batch_label_class.squeeze(),f'preds_reg{dataloader_idx}':conc_pred,f'targets_reg{dataloader_idx}':batch_conc}
        
          
    
    def test_epoch_end(self, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]) -> None:
        # Log individual results for each dataset
        
        for i  in range(len(outputs)):
            dataset_outputs = outputs[i]
            class_preds = torch.cat([x[f'preds_class{i}'] for x in dataset_outputs])
            class_targets = torch.cat([x[f'targets_class{i}'] for x in dataset_outputs])
            conf_mat = MulticlassConfusionMatrix(num_classes=self.hparams.n_class).to("cuda")
            conf_vals = conf_mat(class_preds, class_targets)
            fig = sns.heatmap(conf_vals.cpu() , annot=True, cmap="Blues", fmt="d")
            wandb.log({f"conf_mat{i}" : wandb.Image(fig)})

            reg_preds = torch.cat([x[f'preds_reg{i}'] for x in dataset_outputs])
            reg_targets = torch.cat([x[f'targets_reg{i}'] for x in dataset_outputs])
            data = [[x, y] for (x, y) in zip(reg_targets, reg_preds)]
            table = wandb.Table(data=data, columns = ["True concentration", "Predicted Concentration"])
            wandb.log({f"concentration{i}" : wandb.plot.scatter(table,
                                "True concentration", "Predicted Concentration")})
        return super().test_epoch_end(outputs)
    

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        parser.add_argument('--attn_head',type=int,default=ATT_HEAD)
        parser.add_argument('--encoder_layers',type=int,default=ENCODE_LAYERS)
        parser.add_argument('--n_class',type=int,default=1)
        return parser


def train_pesticide_classifier():
    pl.seed_everything(42, workers=True)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = SERSClassifyRegress.add_model_specific_args(parser)
    parser.add_argument('--num_gpus', type=int, default=AVAIL_GPUS,
                        help="Number of GPUs to use (e.g. -1 = all available GPUs)")
    parser.add_argument('--nodes', type=int, default=NUM_NODES, help="Number of nodes to use")
    parser.add_argument('--num_epochs', type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument('--batch_size', default=BATCH_SIZE, type=int,
                        help="effective_batch_size = batch_size * num_gpus * num_nodes")
    parser.add_argument('--num_dataloader_workers', type=int, default=DATALOADERS)
    parser.add_argument('--entity_name', type=str, default='aghktb', help="Weights and Biases entity name")
    parser.add_argument('--project_name', type=str, default='SERSClassifyRegress',
                        help="Weights and Biases project name")
    parser.add_argument('--save_dir', type=str, default=CHECKPOINT_PATH, help="Directory in which to save models")

    parser.add_argument('--unit_test', type=int, default=False,
                        help="helps in debug, this touches all the parts of code."
                             "Enter True or num of batch you want to send, " "eg. 1 or 7")
    args = parser.parse_args()
    
    args.devices = args.num_gpus
    args.num_nodes = args.nodes
    args.accelerator = ACCELERATOR
    args.max_epochs = args.num_epochs
    args.fast_dev_run = args.unit_test
    args.log_every_n_steps = 1
    args.detect_anomaly = True
    args.enable_model_summary = True
    args.weights_summary = "full"
    
    save_PATH = DATASET_DIR+"/Training2/"+args.save_dir
    os.makedirs(save_PATH, exist_ok=True)

    dataset = SERSDatav2(DATASET_DIR)
    dataset_test2 = SERSDatav2test(DATASET_DIR)
    print(len(dataset))
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - (train_size+val_size)
    dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
    
    # using validation data for testing here
    train_loader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_dataloader_workers)
    print(train_size)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
    test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
    test_loader2 = DataLoader(dataset=dataset_test2,batch_size=BATCH_SIZE,shuffle=False,num_workers=args.num_dataloader_workers)
   # torch.save(test_loader,DATASET_DIR+'/test.pt')
    model = SERSClassifyRegress(learning_rate=1e-4,n_class=Num_classes,attn_head=args.attn_head,encoder_layers=args.encoder_layers)
    
    trainer = pl.Trainer(deterministic=True).from_argparse_args(args)
    
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_top_k=10, dirpath=save_PATH, filename='pesticides_classify_{epoch:02d}_{valid_loss:6f}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping_callback = EarlyStopping(monitor='valid_loss', mode='min', min_delta=0.0, patience=10)
    trainer.callbacks = [checkpoint_callback, lr_monitor, early_stopping_callback]
    logger = WandbLogger(project=args.project_name, entity=args.entity_name,name=args.save_dir, offline=False, save_dir=".")
    trainer.logger = logger
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(dataloaders=[test_loader,test_loader2], ckpt_path='best')
    #trainer.test(dataloaders=test_loader2, ckpt_path='best')
   



if __name__ == "__main__":
    train_pesticide_classifier()
 
