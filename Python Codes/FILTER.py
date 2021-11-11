import pandas as pd
import ast 
from csv import DictWriter

df = pd.read_csv('YOUR_ZERO_SHOT_CLASSİFİER_RESULT_CSV')
threshold_list = [0.50]
field_names = ['metin','label','kategori']
out_path = 'trueTopn_false_threshold'
for item in threshold_list:
  t = 0
  data = {
    'metin':[],
    'label':[],
    'kategori':[],
    }
  df2 = pd.DataFrame(data)
  df2.to_csv(out_path+str(item)+'.csv', index = False)

  for i in range(len(df)):
    D1 = ast.literal_eval(df['multi_class_false'][i])
    D2=ast.literal_eval(df['multi_class_true'][i])
    if D2['labels'][0] != df['kategori'][i] or D1['labels'][0] != df['kategori'][i] :# and D1['scores'][0]>= item[0]:
        t+=1
        print(df['metin'][i],df['kategori'][i])
        
        row_dict = {
        'metin':df['metin'][i], 'label':df['label'][i], 'kategori':df['kategori'][i]
        }

        with open(out_path+str(item)+'.csv', 'a+', newline='') as write_obj:
            dict_writer = DictWriter(write_obj, fieldnames=field_names)
            dict_writer.writerow(row_dict)
      
print(t)
