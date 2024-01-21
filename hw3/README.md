# ADL-HW3

## Environment
* Python 3.10
* PyTorch 2.1.0

***If you don't set up the environment, you can run the following command to set up the environment***

1. Activate conda-env with Python 3.9
```bash
conda create -n adl-hw3 python=3.10
conda activate adl-hw3
```
2. Install Pytorch 2.1.0 with CUDA 11.8
```bash
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
```

3. Install other packages
```bash
pip install -r requirements.txt
```

## Finetune
use the following command to fine-tune the model
```bash
bash finetune.sh
```
**Make sure you put the correct path of dataset and eval_dataser in the script.**

If you want to finetune with Prefix-tuning, add the argument `--prefix_tuning` in the script. (but this requires transformers>4.34.1)

If you want to finetune with P-tuning, add the argument `--P_tuing` in the script. (but this requires transformers>4.34.1)

If you want to finetune with LORA, just make sure `--prefix_tuning` and `--P_tuing` are not in the script.


