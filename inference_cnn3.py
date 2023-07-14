from CLEAN.infer_esm2_cnn3 import *
from CLEAN.utils import *
import csv
import pandas as pd

train_data = "split100"
test_data_price = "price"
test_data_new = "new"

#retrive_esm2_embedding(test_data_new)
#retrive_esm2_embedding(test_data_price)

'''
print("----------------------price-----------------------price--------------------------------")
print(">>>>>>>>>>>>>>>>>>>>>>maxsep>>>>>>>>>>>>>>>>>>>>>>maxsep>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
infer_maxsep(train_data, test_data_price, report_metrics=True, pretrained=False, model_name="split100_supconH")
print(">>>>>>>>>>>>>>>>>>>>>>pvalue>>>>>>>>>>>>>>>>>>>>>>pvalue>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
infer_pvalue(train_data, test_data_price, p_value=1e-5, nk_random=20, report_metrics=True, pretrained=False, model_name="split100_supconH")

print("---------------------new----------------------------new--------------------------------")
print(">>>>>>>>>>>>>>>>>>>>>maxsep>>>>>>>>>>>>>>>>>>>>>>>>>maxsep>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
infer_maxsep(train_data, test_data_new, report_metrics=True, pretrained=False, model_name="split100_supconH")
print(">>>>>>>>>>>>>>>>>>>>>pvalue>>>>>>>>>>>>>>>>>>>>>>>>>pvalue>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
infer_pvalue(train_data, test_data_new, p_value=1e-5, nk_random=20, report_metrics=True, pretrained=False, model_name="split100_supconH")
'''

'''
pre, rec, f1, auc, acc = infer_pvalue(train_data, test_data_new, p_value=1e-5, nk_random=20, report_metrics=True, pretrained=False, model_name="split100_triplet_esm2_7000")
new_pvalue = {'precision':round(pre, 3), 'recall': round(rec, 3), 'F1': round(f1, 3),'AUC': round(auc, 3), 'accuracy':round(acc, 3)}
print(new_pvalue)
'''


def createCSV():
    
    f_new_pvalue = open('./results_esm2/cnn3_new_pvalue.csv', mode='a', encoding='utf-8', newline='')
    new_pvalue_writer= csv.DictWriter(f_new_pvalue, fieldnames=['epoch-new-pvalue', 'precision', 'recall', 'F1', 'AUC', 'accuracy'])
    new_pvalue_writer.writeheader()
    oo = {'epoch-new-pvalue':'cnn3', 'precision':'cnn3', 'recall': 'cnn3', 'F1': 'cnn3','AUC': 'cnn3', 'accuracy':'cnn3'}
    new_pvalue_writer.writerow(oo)
    a = {'epoch-new-pvalue':'论文汇报new', 'precision':'0.5965', 'recall': '0.4811', 'F1': '0.4988','AUC': '0.7399', 'accuracy':''}
    new_pvalue_writer.writerow(a)
    b = {'epoch-new-pvalue':'实际triplet-maxsep-0', 'precision':'0.6577', 'recall': '0.4911', 'F1': '0.5239','AUC': '0.7449', 'accuracy':'0.523'}
    new_pvalue_writer.writerow(b)
    c = {'epoch-new-pvalue':'实际supconH-maxsep', 'precision':'0.6034', 'recall': '0.5408', 'F1': '0.5468','AUC': '0.7697', 'accuracy':'0.5587'}
    new_pvalue_writer.writerow(c)
    
    f_new_maxsep = open('./results_esm2/cnn3_new_maxsep.csv', mode='a', encoding='utf-8', newline='')
    new_maxsep_writer= csv.DictWriter(f_new_maxsep, fieldnames=['epoch-new-maxsep', 'precision', 'recall', 'F1', 'AUC', 'accuracy'])
    new_maxsep_writer.writeheader()
    oo = {'epoch-new-maxsep':'cnn3', 'precision':'cnn3', 'recall': 'cnn3', 'F1': 'cnn3','AUC': 'cnn3', 'accuracy':'cnn3'}
    new_maxsep_writer.writerow(oo)
    a2 = {'epoch-new-maxsep':'论文汇报new', 'precision':'0.5965', 'recall': '0.4811', 'F1': '0.4988','AUC': '0.7399', 'accuracy':''}
    new_maxsep_writer.writerow(a2)
    b2 = {'epoch-new-maxsep':'实际triplet-maxsep-0', 'precision':'0.6577', 'recall': '0.4911', 'F1': '0.5239','AUC': '0.7449', 'accuracy':'0.523'}
    new_maxsep_writer.writerow(b2)
    c2 = {'epoch-new-maxsep':'实际supconH-maxsep', 'precision':'0.6034', 'recall': '0.5408', 'F1': '0.5468','AUC': '0.7697', 'accuracy':'0.5587'}
    new_maxsep_writer.writerow(c2)
    
    
    f_price_pvalue = open('./results_esm2/cnn3_price_pvalue.csv', mode='a', encoding='utf-8', newline='')
    price_pvalue_writer= csv.DictWriter(f_price_pvalue, fieldnames=['epoch-price-pvalue', 'precision', 'recall', 'F1', 'AUC', 'accuracy'])
    price_pvalue_writer.writeheader()
    oo = {'epoch-price-pvalue':'cnn3', 'precision':'cnn3', 'recall': 'cnn3', 'F1': 'cnn3','AUC': 'cnn3', 'accuracy':'cnn3'}
    price_pvalue_writer.writerow(oo)
    a3 = {'epoch-price-pvalue':'论文汇报price', 'precision':'0.5844', 'recall': '0.4671', 'F1': '0.4947','AUC': '0.7334', 'accuracy':''}
    price_pvalue_writer.writerow(a3)
    b3 = {'epoch-price-pvalue':'实际triplet-maxsep-3', 'precision':'0.6112', 'recall': '0.4539', 'F1': '0.4896','AUC': '0.7268', 'accuracy':'0.4564'}
    price_pvalue_writer.writerow(b3)
    c3 = {'epoch-price-pvalue':'实际supconH-pvalue', 'precision':'0.573', 'recall': '0.487', 'F1': '0.5','AUC': '0.743', 'accuracy':'0.4765'}
    price_pvalue_writer.writerow(c3)
    
    
    f_price_maxsep = open('./results_esm2/cnn3_price_maxsep.csv', mode='a', encoding='utf-8', newline='')
    price_maxsep_writer= csv.DictWriter(f_price_maxsep, fieldnames=['epoch-price-maxsep', 'precision', 'recall', 'F1', 'AUC', 'accuracy'])
    price_maxsep_writer.writeheader()
    oo = {'epoch-price-maxsep':'cnn3', 'precision':'cnn3', 'recall': 'cnn3', 'F1': 'cnn3','AUC': 'cnn3', 'accuracy':'cnn3'}
    price_maxsep_writer.writerow(oo)
    a4 = {'epoch-price-maxsep':'论文汇报price', 'precision':'0.5844', 'recall': '0.4671', 'F1': '0.4947','AUC': '0.7334', 'accuracy':''}
    price_maxsep_writer.writerow(a4)
    b4 = {'epoch-price-maxsep':'实际triplet-maxsep-3', 'precision':'0.6112', 'recall': '0.4539', 'F1': '0.4896','AUC': '0.7268', 'accuracy':'0.4564'}
    price_maxsep_writer.writerow(b4)
    c4 = {'epoch-price-maxsep':'实际supconH-pvalue', 'precision':'0.573', 'recall': '0.487', 'F1': '0.5','AUC': '0.743', 'accuracy':'0.4765'}
    price_maxsep_writer.writerow(c4)

def inferenceWirte(epoch, model):
    
    train_data = "split100"
    test_data_price = "price"
    test_data_new = "new"
    
    pre, rec, f1, auc, acc = infer_pvalue(train_data, test_data_new, p_value=1e-5, nk_random=20, report_metrics=True, pretrained=False, model_name=model)
    new_pvalue = [epoch, round(pre, 4), round(rec, 4), round(f1, 4), round(auc, 4), round(acc, 4)]
    print(new_pvalue)
    pd.DataFrame([new_pvalue]).to_csv('./results_esm2/cnn3_new_pvalue.csv',mode='a', header=False, index=False)
    
    pre, rec, f1, auc, acc = infer_maxsep(train_data, test_data_new, report_metrics=True, pretrained=False, model_name=model)
    new_maxsep = [epoch, round(pre, 4), round(rec, 4), round(f1, 4), round(auc, 4), round(acc, 4)]
    print(new_maxsep)
    pd.DataFrame([new_maxsep]).to_csv('./results_esm2/cnn3_new_maxsep.csv',mode='a', header=False, index=False)
    
    pre, rec, f1, auc, acc = infer_pvalue(train_data, test_data_price, p_value=1e-5, nk_random=20, report_metrics=True, pretrained=False, model_name=model)
    price_pvalue = [epoch, round(pre, 4), round(rec, 4), round(f1, 4), round(auc, 4), round(acc, 4)]
    print(price_pvalue)    
    pd.DataFrame([price_pvalue]).to_csv('./results_esm2/cnn3_price_pvalue.csv',mode='a', header=False, index=False)
 
        
    pre, rec, f1, auc, acc = infer_maxsep(train_data, test_data_price, report_metrics=True, pretrained=False, model_name=model)
    price_maxsep = [epoch, round(pre, 4), round(rec, 4), round(f1, 4), round(auc, 4), round(acc, 4)]
    print(price_maxsep)
    pd.DataFrame([price_maxsep]).to_csv('./results_esm2/cnn3_price_maxsep.csv',mode='a', header=False, index=False)

createCSV()
for epoch in range(1000, 15001, 1000):
    inferenceWirte(epoch, 'split100_triplet_cnn3_'+str(epoch))
     
    

'''
#esm1 模型结果 acc
pre, rec, f1, auc, acc = infer_maxsep(train_data, test_data_new, report_metrics=True, pretrained=False, model_name="split100_triplet")
new_maxsep1 = {'epoch':epoch, 'precision':round(pre, 4), 'recall': round(rec, 4), 'F1': round(f1, 4),'AUC': round(auc, 4), 'accuracy':round(acc, 4)}
print(new_maxsep1)

#pre, rec, f1, auc, acc = infer_maxsep(train_data, test_data_new, report_metrics=True, pretrained=False, model_name="split100_supconH_5220")
#new_maxsep1 = {'epoch':epoch, 'precision':round(pre, 4), 'recall': round(rec, 4), 'F1': round(f1, 4),'AUC': round(auc, 4), 'accuracy':round(acc, 4)}
#print(new_maxsep1)

pre, rec, f1, auc, acc = infer_maxsep(train_data, test_data_price, report_metrics=True, pretrained=False, model_name="split100_triplet3")
price_maxsep = {'epoch':epoch, 'precision':round(pre, 4), 'recall': round(rec, 4), 'F1': round(f1, 4),'AUC': round(auc, 4), 'accuracy':round(acc, 4)}
print(price_maxsep)   
 
#pre, rec, f1, auc, acc = infer_pvalue(train_data, test_data_price, p_value=1e-5, nk_random=20, report_metrics=True, pretrained=False, model_name="split100_supconH_5220")
#price_pvalue = {'epoch':epoch, 'precision':round(pre, 4), 'recall': round(rec, 4), 'F1': round(f1, 4),'AUC': round(auc, 4), 'accuracy':round(acc, 4)}
#print(price_pvalue)    
'''
'''

epoch=1000
pre, rec, f1, auc, acc = infer_pvalue(train_data, test_data_new, p_value=1e-5, nk_random=20, report_metrics=True, pretrained=False, model_name="split100_triplet_cnn1_bestloss")
new_pvalue = {'epoch-new-pvalue':epoch, 'precision':round(pre, 4), 'recall': round(rec, 4), 'F1': round(f1, 4),'AUC': round(auc, 4), 'accuracy':round(acc, 4)}
print(new_pvalue)
    
pre, rec, f1, auc, acc = infer_maxsep(train_data, test_data_new, report_metrics=True, pretrained=False, model_name="split100_triplet_cnn1_bestloss")
new_maxsep = {'epoch-new-maxsep':epoch, 'precision':round(pre, 4), 'recall': round(rec, 4), 'F1': round(f1, 4),'AUC': round(auc, 4), 'accuracy':round(acc, 4)}
print(new_maxsep)
    
pre, rec, f1, auc, acc = infer_pvalue(train_data, test_data_price, p_value=1e-5, nk_random=20, report_metrics=True, pretrained=False, model_name="split100_triplet_cnn1_bestloss")
price_pvalue = {'epoch-price-pvalue':epoch, 'precision':round(pre, 4), 'recall': round(rec, 4), 'F1': round(f1, 4),'AUC': round(auc, 4), 'accuracy':round(acc, 4)}
print(price_pvalue)    
 
        
pre, rec, f1, auc, acc = infer_maxsep(train_data, test_data_price, report_metrics=True, pretrained=False, model_name="split100_triplet_cnn1_bestloss")
price_maxsep = {'epoch-price-maxsep':epoch, 'precision':round(pre, 4), 'recall': round(rec, 4), 'F1': round(f1, 4),'AUC': round(auc, 4), 'accuracy':round(acc, 4)}
print(price_maxsep)
'''
    