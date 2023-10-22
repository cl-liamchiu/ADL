# ADL-HW1

## Environment
* Python 3.9 and Python Standard Library
* PyTorch 1.12.1, scikit-learn 1.1.2, nltk 3.7
* tqdm, numpy, pandas
* transformers 4.22.2, datasets 2.5.2, accelerate 0.13.0
* evaluate, matplotlib, gdown

***If you don't set up the environment, you can run the following command to set up the environment***
```bash
cd code_and_script
pip install -r requirements.txt
cd .. # to r12922a14/
```
## Run the code
1. Download models, tokenizers and data

```bash
bash ./download.sh
```

If there are some errors, you can download the models, tokenizers and data from [here](https://www.dropbox.com/scl/fi/j3ymlndgtnf1m2fn6mqfj/models_tokenizers_and_data.zip?rlkey=iehttcyfmiltbfiprey7fgbn6&dl=0) and put the zip fule into `r12922a14/` folder and unzip it to `r12922a14/models_tokenizers_and_data/`.

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
    ├── mc_model/
    ├── qa_model/
 	└── qa_model_end_to_end/
```

3. Predict testing data
```bash
bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv
```

4. Train the model
```bash
cd code_and_script

# train multiple choice model (Paragraph Selection) 
bash ./train_mc.sh /path/to/context.json /path/to/train.json /path/to/validation.json /path/to/saved/model_dir/

# train question answering model (Span Selection (Extractive QA))
bash ./train_qa.sh /path/to/context.json /path/to/train.json /path/to/validation.json /path/to/saved/model_dir/

# train question answering model (Span Selection (Extractive QA)) with recording loss and exact match
bash ./train_qa_record_loss.sh /path/to/context.json /path/to/train.json /path/to/validation.json /path/to/saved/model_dir/

# train question answering model (Span Selection (Extractive QA)) from scratch
bash ./train_qa_from_scratch.sh /path/to/context.json /path/to/train.json /path/to/validation.json /path/to/saved/model_dir/

# train question answering model (Span Selection (Extractive QA)) end to end
bash ./train_qa_end_to_end.sh /path/to/context.json /path/to/train.json /path/to/validation.json /path/to/saved/model_dir/
```

5. Plot the loss and exact match
```bash
python plot_report_figures.py --log_file /path/to/log_file --output_dir /path/to/figure_dir/

# if you train the model with recording loss and exact match, you will get the log file in the model_dir
``` 