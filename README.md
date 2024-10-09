# MMSite: A Multi-modal Framework for the Identification of Active Sites in Proteins
## 1. Preparation
### 1. Environment
You can manage the environment by Anaconda. We have provided the environment configuration file `environment.yml` for reference. You can create the environment by the following command:
```bash
conda env create -f environment.yml
```
### 2. Data
You can follow the instructions in `dataprocess/README.md` to prepare the data. In this `.md` file, we provide the instruction to split the data when the clustering threshold is 10%. You can also change the threshold when you execute the mmseqs2 command.
## 2. Training
### 2.1 Download the Pre-trained Model
In our MMSite, we use the pre-trained PLM and BLM models to initialize the features. You can download the pre-trained model from the Higging Face to reproduce the main results in our paper. You can put all the downloaded models in the `pretrained_weights` folder.
### 2.1 Configuration
You can specify the configuration in `config.yaml`, including the paths of the pre-trained models and the data, training parameters, etc.
### 2.2 Training
You can train the model by the following command (It takes about 7 hours to finish training on a single NVIDIA GeForce RTX 4090 GPU):
```bash
python train.py --config /path/to/config.yaml
```
Then, you will get `best_model_fuse_xxx.pth` model in the `runs/timestamp` folder, which is the final model.
## 3. Inference
You should put your data in the `dataset/infer.tsv` with the format like `dataset/infer_samples.tsv`. Then, you should specify the path of `best_model_fuse_xxx.pth` in inference.py. Finally, you can run the following command to get the prediction results:
```bash
python inference.py
```
