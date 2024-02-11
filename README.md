# SERSFormer : A Machine Learning Approach for Rapid Detection of Pesticides by SERS Coupled with Transformers

![SERSFormer](./blockdiag.png?raw=true "SERSFormer Architecture Diagram")

The SERSFormer, is a multi-tasking weight sharing transformer based model, designed for identifying and quantifying the pesticide residue present on the foodsample. It takes SERS spectrum of food sample as input and performs two tasks- classification and regression simultaneously. The above block diagram shows the multi-tasking architecture of SERSFormer. 

The repository contains the SERS dataspectra for 5 different pesticides that are commonly found on spinach, thiabendazole, phosmet, coumaphos, carbophenothion and oxamyl respectiveely and a control sam[le without any pesticides. Each pesticide contains, 5 different concentration ranges from 0 tp 10 ppm.]

To use this repository, clone the repository to required folder on your system using 

`git clone https://github.com/BioinfoMachineLearning/SERSFormer.git`

set up conda environement and install necessary packages using the setup.sh script.

```
cd SERSFormer
./setup.sh 
```
To train the model, validate and test, run the following command:
```
python SERSFormer_Training.py \
--attn_head 4 \
--encoder_layers 6\
--save_dir SERSFormer_log\
--entity_name YourWandbUserName 
```
SERSFormer uses Wandb for logging all the metrics and training parameters. Provide wandb login username in the arguement to monitor training in realtime. It can be customized to log any media, text, images, graphs, gradients, and metrics. For more information on setting up wandb, please visit the documentation https://docs.wandb.ai/guides/integrations/lightning

**Cite Us**

If this repository is useful, please cite us.

Hajikhani, M., Hegde, A., Snyder, J., Cheng, J., & Lin, M. (2024). A Machine Learning Approach for Rapid Detection of Pesticides by SERS Coupled with Transformers.  https://github.com/BioinfoMachineLearning/SERSFormer.git


