from sklearn.model_selection import train_test_split
import sklearn
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification,pipeline

import re
from simpletransformers.classification import ClassificationModel

data = pd.read_csv('data set path according to which method you will use ')
HUGGINGFACE_MODEL_PATH = "loodos/bert-base-turkish-uncased"
MODEL_OUTPUT_DIR = 'BERT4_no_zero_shot_44/'

#data = data[data.groupby('kategori')['kategori'].transform('size') > 200]
#data = data.reset_index(drop = True)

unique_list = data['label'].unique().tolist()

def replace_with_index(x):
  index = unique_list.index(x)
  return index


data['label'] = data['label'].apply(lambda x: replace_with_index(x))

data = data.iloc[:,:2]

for i in range(len(data['label'])):
  if isinstance(data['label'][i], np.generic):
    data['label'][i]= np.asscalar(data['label'][i])

data.dropna(axis= 0, inplace=True)
data = data.reset_index(drop = True)

def cleanText(input_sentence):
 
  tmp= [word.replace('A','a') for word in input_sentence.split(' ')]
  tmp= [word.lower() for word in tmp]
  tmp= [word.replace('i̇','i') for word in tmp]
  tmp = [re.sub('[^A-Za-z0-9ğüşıçöiâî]+', ' ', word) for word in tmp]
  tmp = [word.strip(' ') for word in tmp]
  tmp1 =' '.join(tmp)

  return tmp1

data["metin"] = data["metin"].apply(cleanText)
print(len(data))
data = data.drop_duplicates()
print(len(data))
data=data.sample(frac=1)
data = data.reset_index(drop= True)


#print(data['label'].unique())

X_train, X_test, y_train, y_test = train_test_split(data["metin"],data["label"], test_size= .2, stratify=data['label'], random_state = 42)
train = pd.concat([X_train,y_train], axis = 1)

tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)
model = AutoModel.from_pretrained(HUGGINGFACE_MODEL_PATH)
no_labels = len(set(data["label"]))


model_args = {
    "use_early_stopping": True,
    "early_stopping_delta": 0.01,
    "early_stopping_metric": "mcc",
    "early_stopping_metric_minimize": False,
    "early_stopping_patience": 5,
    "evaluate_during_training_steps": 6000,
    "fp16": False,
    "num_train_epochs":3
}

model = ClassificationModel(
    "bert", 
    HUGGINGFACE_MODEL_PATH,
     use_cuda=True, 
     args=model_args, 
     num_labels=no_labels
)

model.train_model(train, acc=sklearn.metrics.accuracy_score, output_dir=MODEL_OUTPUT_DIR)
EPOCH_PATH = [MODEL_OUTPUT_DIR +"/"+ i for i in os.listdir(MODEL_OUTPUT_DIR) if "epoch" in i and "-3"in i]
EPOCH_PATH=EPOCH_PATH[0]

def get_model(path):

  result = {}
  plt.rcParams["figure.figsize"] = (50,30)
  for epoch in path:

    tokenizer= AutoTokenizer.from_pretrained(EPOCH_PATH)

    # build and load model, it take time depending on your internet connection
    model= AutoModelForSequenceClassification.from_pretrained(EPOCH_PATH)

    # make pipeline
    nlp=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


    
    test = pd.concat([X_test,y_test],axis=1)
    print(test)
    #test_unique_list = test['label'].unique().tolist()
    #test_unique_list.sort()
    #print(test_unique_list)
    pred_label_list, true_label_list = [],[]
    cou = 0
    for t,l in zip(test["metin"], test["label"]):
      cou+=1
      true_label_list.append(l)
      pred_label_list.append(int(nlp(t)[0]["label"].lstrip("LABEL_")))

      if cou %1000 == 0:
        print(EPOCH_PATH.index(epoch),cou)

    df = pd.DataFrame()
    df["true"] = true_label_list
    df["pred"] = pred_label_list
    print(df)

    """for item in test_unique_list:
      true_df = df[df.true == item]
      class_acc = accuracy_score(true_df["true"], true_df["pred"])
      print('ID = {} ACC = {}'.format(item, class_acc))"""
    

    acc = accuracy_score(df["true"], df["pred"])
    result[f"{EPOCH_PATH.index(epoch)+1}. epoch Accuracy"] = acc
    print(acc)
    print(classification_report(df["true"], df["pred"]))

    sns.heatmap(confusion_matrix(df["true"], df["pred"]), annot = True, fmt = "g")
    plt.show()
    return df

data_test = pd.DataFrame()
data_test = get_model(EPOCH_PATH)
