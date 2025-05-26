# Privacy-Preserving and Robust Black-Box Knowledge Distillation

This repository contains the code for my project of Trustworthy Machine Learning.

For more details, please refer to 'Aligning Logits Generatively for Principled Black-Box Knowledge Distillation' [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Aligning_Logits_Generatively_for_Principled_Black-Box_Knowledge_Distillation_CVPR_2024_paper.html).


## Environment

This implementation is based on PyTorch, and we recommend using a GPU to run the experiments.

Install the required packages using `conda` and `pip`:

```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```


## Run

### 1. Prepare a Teacher Model

If you don't already have a teacher model ready, you'll need to train one.

```shell
python -u project/teacher.py --model ResNet32

# or choose to run in the background monitor and output the logs
nohup python -u project/teacher.py --model ResNet32 > output.log 2>error.log &
```

### 2. Deprivatization

The first step is to train a DCGAN to accomplish the deprivatization, as mentioned in our our [paper](https://openaccess.thecvf.com/content/CVPR2024/html/Ma_Aligning_Logits_Generatively_for_Principled_Black-Box_Knowledge_Distillation_CVPR_2024_paper.html).


```shell
python -u project/DCGAN.py --teacher ResNet32

# or choose to run in the background monitor and output the logs
nohup python -u project/DCGAN.py --teacher ResNet32 > output.log 2>error.log &
```

### 3. Distillation

The second step carries out black-box knowledge distillation, and you can experiment with different KD methods by changing the `{method}` parameter.

```shell
python -u project/{method}.py --teacher ResNet32 --model ResNet8

# or choose to run in the background monitor and output the logs
nohup python -u project/{method}.py --teacher ResNet32 --model ResNet8 > output.log 2>error.log &
```

`{method}` can be one of the following:
`KD`, `ML`, `AL`, `DKD`, `DAFL`, `KN`, `AM`, `DB3KD`, `MEKD`.

For `MEKD`, you can specify the `--res_type` parameter to choose the response type, which can be:
`soft`, `hard`.
