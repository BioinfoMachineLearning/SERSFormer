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
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import torch.nn as nn
import numpy as np
from pandas import read_csv, DataFrame
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
from torchmetrics.regression import MeanSquaredError,R2Score,MeanAbsoluteError,SpearmanCorrCoef
from scipy.stats import gaussian_kde
from Model_v2 import TransformerClassifyRegress_con,TransformerClassifyRegress_sep
from argparse import ArgumentParser
import scipy.signal as signal
from Dataset import SERSDatav2,SERSDatav3,SERSDatav3test
AVAIL_GPUS = [0,1,2]
NUM_NODES = 1
BATCH_SIZE = 32
DATALOADERS = 1
ACCELERATOR = 'gpu'
EPOCHS = 40
ATT_HEAD = 4
ENCODE_LAYERS = 4
DATASET_DIR = "/home/aghktb/AI-SERS"

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
        self.num_outputs = 1
        self.save_hyperparameters()
        
        self.model = TransformerClassifyRegress_sep(attn_head=attn_head,encoder_layers=encoder_layers,n_class=n_class,**model_kwargs)
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
               'monitor':metric_to_track,
               'watch':self.logger.watch(self.model)}
    
    def custom_viz(self,kernels, path=None, cols=None):
        """Visualize weight and activation matrices learned 
        during the optimization process. Works for any size of kernels.
        
        Arguments
        =========
        kernels: Weight or activation matrix. Must be a high dimensional
        Numpy array. Tensors will not work.
        path: Path to save the visualizations.
        cols: TODO: Number of columns (doesn't work completely yet.)
        
        Example
        =======
        kernels = model.conv1.weight.cpu().detach().clone()
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        custom_viz(kernels, 'results/conv1_weights.png', 5)
        """
        def set_size(w,h, ax=None):
            """ w, h: width, height in inches """
            if not ax: ax=plt.gca()
            l = ax.figure.subplotpars.left
            r = ax.figure.subplotpars.right
            t = ax.figure.subplotpars.top
            b = ax.figure.subplotpars.bottom
            figw = float(w)/(r-l)
            figh = float(h)/(t-b)
            ax.figure.set_size_inches(figw, figh)
    
        N = kernels.shape[0]
        C = kernels.shape[1]

        Tot = N*C

    # If single channel kernel with HxW size,# plot them in a row.# Else, plot image with C number of columns.
        if C>1:
            columns = C
        elif cols==None:
            columns = N
        elif cols:
            columns = cols
        rows = Tot // columns 
        rows += Tot % columns

        pos = range(1,Tot + 1)

        fig = plt.figure(1)
        fig.tight_layout()
        k=0
        for i in range(kernels.shape[0]):
            for j in range(kernels.shape[1]):
                img = kernels[i][j]
                ax = fig.add_subplot(rows,columns,pos[k])
                #ax.imshow(img, cmap='gray')
                plt.axis('off')
                k = k+1

        set_size(30,30,ax)
        if path:
            plt.savefig(path, dpi=100)
    
        #plt.show()
        return fig
    
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
        
       
    
    def test_step(self,batch, batch_idx):
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
       
        
        return {f'preds_class' : class_pred, f'targets_class' : batch_label_class.squeeze(),f'preds_reg':conc_pred,f'targets_reg':batch_conc}
        
          
    
    def test_epoch_end(self, outputs: EPOCH_OUTPUT | List[EPOCH_OUTPUT]) -> None:
        # Log individual results for each dataset
        
        #for i  in range(len(outputs)):
            dataset_outputs = outputs
            torch.save(dataset_outputs,"Predictions.pt")
            class_preds = torch.cat([x[f'preds_class'] for x in dataset_outputs])
            class_targets = torch.cat([x[f'targets_class'] for x in dataset_outputs])
            conf_mat = MulticlassConfusionMatrix(num_classes=Num_classes)
            conf_vals = conf_mat(class_preds, class_targets)
            fig, ax = plt.subplots()
            sns.heatmap(conf_vals.cpu() , annot=True, cmap="Blues", fmt="d")
            plt.title("Confusion Matrix")
            plt.tight_layout()
            wandb.log({f"Confusion Matrix" :wandb.Image(fig)})
            reg_preds = torch.cat([x[f'preds_reg'] for x in dataset_outputs])
            reg_targets = torch.cat([x[f'targets_reg'] for x in dataset_outputs])
            data = [[x, y] for (x, y) in zip(reg_targets, reg_preds)]
            reg_preds_np = reg_preds.squeeze().numpy()
            reg_targets_np = reg_targets.squeeze().numpy()

            df = DataFrame({'True': reg_targets_np, 'Pred': reg_preds_np})

            # Scatter plot
            # Calculate the point density
            xy = torch.vstack([reg_targets.T,reg_preds.T]).cpu().detach().numpy()
            
            z = gaussian_kde(xy.squeeze())(xy.squeeze())
            # Sort the points by density, so that the densest points are plotted last
            idx = z.argsort()
            x, y, z = reg_targets[idx], reg_preds[idx], z[idx]
            fig1, ax = plt.subplots(figsize=(12, 6))
            plt.scatter(x,y,c=z,s=100)
            wandb.log({f"Concentration Predictions" :wandb.Image(fig1)})
            # Get unique target values
            unique_targets = df['True'].unique()
            unique_targets = sorted(unique_targets)
            print(unique_targets)
            # Set up subplots
            num_targets = len(unique_targets)
            fig2, axes = plt.subplots(nrows=1, ncols=num_targets, figsize=(15, 5),sharey=True)

            # Create violin plots for each target value
            for i, target_value in enumerate(unique_targets):
                target_df = df[df['True'] == target_value]
                sns.violinplot(x='True', y='Pred', data=target_df, inner="sticks", color="lightgreen",cut=0 ,ax=axes[i])

                axes[i].set_title(f'Target = {target_value}')

            plt.tight_layout()
            plt.xlabel("True Concentration")
            plt.ylabel("Predicted Concentration")
            plt.title("True and Predicted Concentration")
            plt.tight_layout()
            wandb.log({f"Kernel Density Estimation of Concentrations" :wandb.Image(fig2)})
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

    dataset = SERSDatav3(DATASET_DIR)
    #dataset_test = SERSDatav3test(DATASET_DIR)
    print(len(dataset))
    train_size = int(0.7 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - (train_size+val_size)
    dataset_train,dataset_valid,dataset_test = torch.utils.data.random_split(dataset, [train_size, val_size,test_size])
    
    #dataset_valid = MicrographDataValid(DATASET_DIR)
    #dataset_test = MicrographDataValid(DATASET_DIR) # using validation data for testing here
    train_loader = DataLoader(dataset=dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=args.num_dataloader_workers)
    print(train_size)
    valid_loader = DataLoader(dataset=dataset_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
    test_loader = DataLoader(dataset=dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=args.num_dataloader_workers)
    #test_loader2 = DataLoader(dataset=dataset_test2,batch_size=BATCH_SIZE,shuffle=False,num_workers=args.num_dataloader_workers)
    torch.save(test_loader,DATASET_DIR+'/'+args.save_dir+'_test.pt')
    model = SERSClassifyRegress(learning_rate=1e-4,n_class=Num_classes,attn_head=args.attn_head,encoder_layers=args.encoder_layers)
    
    trainer = pl.Trainer().from_argparse_args(args)
    
    checkpoint_callback = ModelCheckpoint(monitor='valid_loss', save_top_k=10, dirpath=save_PATH, filename='pesticides_classify_{epoch:02d}_{valid_loss:6f}')
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    early_stopping_callback = EarlyStopping(monitor='valid_loss', mode='min', min_delta=0.0, patience=10)
    trainer.callbacks = [checkpoint_callback, lr_monitor, early_stopping_callback]
    
    #wandb.init(project=args.project_name, entity=args.entity_name,name=args.save_dir,sync_tensorboard=True).watch(model)
    logger = WandbLogger(project=args.project_name, entity=args.entity_name,name=args.save_dir, offline=False, save_dir=".",sync_tensorboard=True)
    logger1 = TensorBoardLogger(save_dir=".",name=args.save_dir)
    trainer.logger = logger
    
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(dataloaders=test_loader, ckpt_path='best')
    
    #trainer.test(dataloaders=test_loader2, ckpt_path='best')
   



if __name__ == "__main__":
    
    train_pesticide_classifier()
    #wandb.finish()

    '''predict_class = [p[0].item() for p in predict]
    predict_reg = [p[1].item() for p in predict]
    targets_class = list()
    targets_reg = []
    
    for i in range(len(dataset_test)):
        pest_datas = dataset_test[i]
        pest_data = pest_datas[2]
        pest_class_label = pest_datas[0]
        pest_reg_label = pest_datas[1]
        targets_class.append(pest_class_label)
        targets_reg.append(pest_reg_label)
    metrics_class = MetricCollection([MulticlassAccuracy(num_classes=Num_classes),
                                         MulticlassPrecision(num_classes=Num_classes),
                                         MulticlassRecall(num_classes=Num_classes),
                                         MulticlassF1Score(num_classes=Num_classes)])
    metrics_regress = MetricCollection([ MeanSquaredError(),
                                                 R2Score(),
                                                 MeanAbsoluteError()])
    metric_log_class = metrics_class(torch.tensor(predict_class), torch.tensor(targets_class).squeeze())
    print(metric_log_class)
    metric_log_reg = metrics_class(torch.tensor(predict_reg), torch.tensor(targets_reg))
    print(metric_log_reg)

    bcm = MulticlassConfusionMatrix(num_classes=Num_classes)
    # Generate the confusion matrix
    cm = bcm(torch.tensor(predict), torch.tensor(targets_class).squeeze())

    # Create the heatmap of the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("eval.png")
    # plt.show()'''
