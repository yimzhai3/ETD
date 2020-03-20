# Enhanced Transport Distance for Unsupervised Domain Adaptation (ETD)
This is the `pytorch` demo code for Enhanced Transport Distance for Unsupervised Domain Adaptation (ETD) (CVPR 2020)
## Requirements
* python 3.7
* torch 1.2.0
* torchvision 0.4.0
* pandas 0.24.2
* numpy 1.17.3
## Dataset

* The structure of the datasets should be like
```
OfficeHome (Dataset)
|- Art (Domain)
|  |- Alarm_Clock (Class)
|     |- 00001.jpg (Sample) 
|     |- ...
|  |- Backpack (Class)
|  |- ...
|- Clipart
|- Product 
|- Real_World

```
* The srtucture of all the code should be like
```
|- OfficeHome
|- UV_code
```
## Usage
* Download the `OfficeHome` dataset from Google Drive.
* Set experiment configures in a csv file.
  * It is `UV.csv` in this code.
  * The csv includes: epochs, Pretrain_Epoch, train_batch_size, lr	lr_feature, lr_fc, beta1, beta2, lambda_1, lambda_2, source_domain, target_domain, class_num, resnet_name, fc_in_features, bottleneck_dim, dropout_p, and network_name.
  * An example
* Set saving path.
  * The saving path is `./UV_code/UV` in this code and the corresponding code is as following:
  
  ```
   file_path = '.'+os.path.sep+'UV' 
   if not os.path.exists(file_path):
      os.mkdir(file_path)
   experiment_base_path = '.'+os.path.sep+'UV'+os.path.sep+experiment_name        
   if not os.path.exists(experiment_base_path):
      os.mkdir(experiment_base_path)
  ```
 * Training with `main.py`.
 * The loss, acc, best acc and best model can be found in `./UV_code/UV/test1`(in this code). 
# Note
This code is correspongding to the dual formulation of the reweighed OT problem. And we will introduce the semi-dual version later. 
## Citation
```
If this reposity is helpful for you, please cite our paper:
@inproceedings{Li2020ETD,
  title={Enhanced Transport Distance for Unsupervised Domain Adaptation},
  author={Mengxue Li, and Yi-Ming Zhai, and You-Wei Luo, and Peng-Fei Ge, and Chuan-Xian Ren},
  booktitle={2020 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2020},
}
```
## Contact
If you have any questions, please feel free to contact me via **zhaiym3@mail2.sysu.edu.cn**.
