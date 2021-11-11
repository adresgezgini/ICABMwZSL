from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import os 
import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
from transformers import AutoModel, AutoTokenizer,AutoModelForSequenceClassification,pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def replace_with_index(x):
    index = unique_list.index(x)
    return index

def cleanText(input_sentence):
  
    tmp= [word.replace('A','a') for word in input_sentence.split(' ')]
    tmp= [word.lower() for word in tmp]
    tmp= [word.replace('i̇','i') for word in tmp]
    tmp = [re.sub('[^A-Za-z0-9ğüşıçöiâî]+', ' ', word) for word in tmp]
    tmp = [word.strip(' ') for word in tmp]
    tmp1 =' '.join(tmp)

    return tmp1

def get_model(path,X_test,y_test):

    result = {}
    plt.rcParams["figure.figsize"] = (50,30)
    for epoch in path:

      tokenizer= AutoTokenizer.from_pretrained(path)

      # build and load model, it take time depending on your internet connection
      model= AutoModelForSequenceClassification.from_pretrained(path)

      # make pipeline
      nlp=pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


      
      test = pd.concat([X_test,y_test],axis=1)
      #print(test)
      test_unique_list = test['label'].unique().tolist()
      test_unique_list.sort()
      #print(test_unique_list)
      pred_label_list, true_label_list = [],[]
      cou = 0
      for t,l in zip(test["metin"], test["label"]):
        cou+=1
        true_label_list.append(l)
        pred_label_list.append(int(nlp(t)[0]["label"].lstrip("LABEL_")))

        if cou %1000 == 0:
          print(path.index(epoch),cou)

      df = pd.DataFrame()
      df["true"] = true_label_list
      df["pred"] = pred_label_list
      print(df)

      for item in test_unique_list:
        true_df = df[df.true == item]
        class_acc = accuracy_score(true_df["true"], true_df["pred"])
        print('ID = {} ACC = {}'.format(item, class_acc),file=open("output.txt", "a"))
      

      acc = accuracy_score(df["true"], df["pred"])
      result[f"{path.index(epoch)+1}. epoch Accuracy"] = acc
      print('acc = {}'.format(acc),file=open("output.txt", "a"))
      print(classification_report(df["true"], df["pred"]),file=open("output.txt", "a"))

      """sns.heatmap(confusion_matrix(df["true"], df["pred"]), annot = True, fmt = "g")
      plt.show()"""
      return df

HUGGINGFACE_MODEL_PATH = "loodos/bert-base-turkish-uncased"
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL_PATH)

print('Zero_Shotsız_ve_Zero_Shotlı_BERT_Result',file=open("output.txt", "a"))
list_url = [['Single_BERT_DATASET_PATH','No_ZEROSHOT'],['FİLTERED_BY_ZEROSHOT_RESULT_DATASET_PATH','W_ZEROSHOT']]

    
for item in list_url:
  print(item[1],file=open("output.txt", "a"))
  data = pd.read_csv(item[0])
  unique_list = data['label'].unique().tolist()
  data['label'] = data['label'].apply(lambda x: replace_with_index(x))
  data = data.iloc[:,:2]

  for i in range(len(data['label'])):
    if isinstance(data['label'][i], np.generic):
      data['label'][i]= np.asscalar(data['label'][i])

  data.dropna(axis= 0, inplace=True)
  data = data.reset_index(drop = True)

  data["metin"] = data["metin"].apply(cleanText)
  print(len(data))

  data = data.drop_duplicates()
  print(len(data))

  data=data.sample(frac=1)
  data = data.reset_index(drop= True)

  train_data = pd.concat([data["metin"],data["label"]], axis = 1)

  # prepare cross validation
  n=5
  kf = StratifiedKFold(n_splits=n, random_state=42, shuffle=True)

  no_labels = len(set(data["label"]))
  results = []
  for i,(train_index, val_index) in enumerate(kf.split(train_data)):
      print('Kesit No: {}'.format(i),file=open("output.txt", "a"))
    # splitting Dataframe (dataset not included)
      train_df = train_data.iloc[train_index]
      val_df = train_data.iloc[val_index]
      # Defining Model
      MODEL_OUTPUT_DIR = 'BERT_{}_44_kategori_{}/'.format(item[1],i,)

      #model = AutoModel.from_pretrained(HUGGINGFACE_MODEL_PATH)
      model_args = {
          "use_early_stopping": True,
          "early_stopping_delta": 0.01,
          "early_stopping_metric": "mcc",
          "early_stopping_metric_minimize": False,
          "early_stopping_patience": 5,
          "evaluate_during_training_steps": 6000,
          "fp16": False,
          "num_train_epochs":3
          #"overwrite_output_dir": True
      }

      model = ClassificationModel(
          "bert", 
          HUGGINGFACE_MODEL_PATH,
          use_cuda=True, 
          args=model_args, 
          num_labels=no_labels
      )

      model.train_model(train_df, acc=accuracy_score, output_dir=MODEL_OUTPUT_DIR)
      EPOCH_PATH = [MODEL_OUTPUT_DIR +"/"+ t for t in os.listdir(MODEL_OUTPUT_DIR) if "epoch" in t and "-3"in t]
      EPOCH_PATH=EPOCH_PATH[0]

      data_test = pd.DataFrame()
      data_test = get_model(EPOCH_PATH,val_df["metin"],val_df["label"])

