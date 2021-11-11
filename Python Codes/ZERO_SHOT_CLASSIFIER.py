from transformers import pipeline
import torch
import pandas as pd
import time 
from csv import DictWriter
device = 0 if torch.cuda.is_available() else -1
field_names = ['metin','label','kategori','multi_class_true','multi_class_false']

data = {
    'metin':[],
    'label':[],
    'kategori':[],
    'multi_class_true':[],
    'multi_class_false':[]
}
df = pd.DataFrame(data)
df.to_csv('Result_Zero_Shot.csv', index = False)

classifier = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=device)
df = pd.read_csv('your_data_csv_format')
labels = df['kategori'].unique()

for i in range(len(df['metin'])):
    input_text1 = df['metin'][i]
    multi_true_result = classifier(input_text1, labels,multi_class=True)
    multi_false_result = classifier(input_text1, labels,multi_class=False)
    row_dict = {
        'metin':df['metin'][i], 'label':df['label'][i], 'kategori':df['kategori'][i], 'multi_class_true':multi_true_result ,'multi_class_false':multi_false_result
        }
    
    with open('result.csv', 'a+', newline='') as write_obj:
        dict_writer = DictWriter(write_obj, fieldnames=field_names)
        dict_writer.writerow(row_dict)
    #print(result)
