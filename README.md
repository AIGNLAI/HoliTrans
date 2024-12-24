## Environment

This repository is tested in an Anaconda environment. To reproduce exactly, create your environment as follows:

```
conda create -y -n UOC python=3.9
conda activate UOC
conda install -y pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -y -c anaconda pandas==1.5.2
pip install tqdm==4.65.0
pip install timm==0.6.12
pip install easydict
```

## To reproduce results run code of the form

CIRO

```
python main.py -i 7 -d cifar224
```

KIRO

```
python main.py -i 7 -d core50
```

