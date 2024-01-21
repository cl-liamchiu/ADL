# ADL-HW2

## Environment
* Python 3.9 and Python Standard Library
* PyTorch 2.1.0

***If you don't set up the environment, you can run the following command to set up the environment***

1. Activate conda-env with Python 3.9
```bash
conda create -n adl-hw2 python=3.9
conda activate adl-hw2
```
2. Install Pytorch 2.1.0 with CUDA 11.8
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3. Install other packages
```bash
cd code_and_script
pip install -r requirements.txt
cd .. # to r12922a14/
```

## Predict testing data
1. Download models, tokenizers and data

```bash
bash ./download.sh
```

If there are some errors, you can download the models, tokenizers and data from [here](https://drive.google.com/file/d/1TElYcYNWtl8Uml0nVfbEKpZlVqIbcEMJ/view) and put the zip fule into `r12922a14/` folder and unzip it to `r12922a14/models_tokenizers_and_data/`.

2. Check folder tree:
```
r12922a14/
├── code_and_script/
├── download.sh
├── README.md
├── report.pdf
├── run.sh
└── models_tokenizers_and_data/
	├── data/
 	└── model/
```

3. Predict testing data
```bash
bash ./run.sh /path/to/input.jsonl /path/to/output.jsonl
```

## Evaluate the model
```bash
cd code_and_script
python eval.py -r public.jsonl -s submission.jsonl
cd .. # to r12922a14/
```

## Train the model
```bash
cd code_and_script
pip install -e tw_rouge
bash ./train.sh
```

