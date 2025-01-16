# Exploring Open-world Continual Learning with Knowns-Unknowns Knowledge Transfer

## Environment

This repository has been tested in an Anaconda environment. To replicate the results precisely, set up your environment using the following steps:

```bash
conda create -y -n HoliTrans python=3.9
conda activate HoliTrans
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c anaconda pandas==1.5.2
pip install tqdm==4.65.0
pip install timm==0.6.12
pip install easydict
```

## Running the Code to Reproduce Results

To reproduce the results, execute commands in the following format:

For **CIRO** (using the CIFAR-224 dataset):

```bash
python main.py -i 7 -d cifar224
```

For **KIRO** (using the Core50 dataset):

```bash
python main.py -i 7 -d core50
```

Replace the dataset flag (`-d`) and other parameters as needed to suit your specific experimental settings.

## Contact

For any questions or further information, please contact the authors:

- Yujie Li: [y.li@liacs.leidenuniv.nl](mailto:y.li@liacs.leidenuniv.nl)
- Guannan Lai: [aignlai@163.com](mailto:aignlai@163.com)

