import torch
import time
import os
import pickle
from CLEAN.dataloader import *
from CLEAN.model import *
from CLEAN.utils import *
from CLEAN.infer import *
import torch.nn as nn
import argparse
from CLEAN.distance_map import get_dist_map
import csv
import pandas as pd
import torch, gc
from torch.cuda.amp import autocast




def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--learning_rate', type=float, default=5e-4)
    parser.add_argument('-e', '--epoch', type=int, default=2000)
    parser.add_argument('-n', '--model_name', type=str, default='split10_triplet')
    parser.add_argument('-t', '--training_data', type=str, default='split10')
    parser.add_argument('-d', '--hidden_dim', type=int, default=512)
    parser.add_argument('-o', '--out_dim', type=int, default=128)
    parser.add_argument('--adaptive_rate', type=int, default=100)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()
    return args


def get_dataloader(dist_map, id_ec, ec_id, args):
    params = {
        #'batch_size': 6000,
        'batch_size': 6000,
        'shuffle': True,
    }
    negative = mine_hard_negative(dist_map, 30)
    train_data = Triplet_dataset_with_mine_EC(id_ec, ec_id, negative)
    train_loader = torch.utils.data.DataLoader(train_data, **params)
    return train_loader


def train(model, args, epoch, train_loader,
          optimizer, device, dtype, criterion):
    model.train()
    total_loss = 0.
    start_time = time.time()

    for batch, data in enumerate(train_loader):
        optimizer.zero_grad()
        anchor, positive, negative = data
        #print(anchor.shape)
        with autocast():
            anchor_out = model(anchor.to(device=device, dtype=dtype))
            positive_out = model(positive.to(device=device, dtype=dtype))
            negative_out = model(negative.to(device=device, dtype=dtype))

            loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if args.verbose:
            lr = args.learning_rate
            ms_per_batch = (time.time() - start_time) * 1000
            cur_loss = total_loss 
            print(f'| epoch {epoch:3d} | {batch:5d}/{len(train_loader):5d} batches | '
                  f'lr {lr:02.4f} | ms/batch {ms_per_batch:6.4f} | '
                  f'loss {cur_loss:5.2f}')
            start_time = time.time()
    # record running average training loss
    return total_loss/(batch + 1)

def createCSV():
    
    f_new_pvalue = open('./results_esm2/cnn3_new_pvalue.csv', mode='a', encoding='utf-8', newline='')
    new_pvalue_writer= csv.DictWriter(f_new_pvalue, fieldnames=['epoch-new-pvalue', 'precision', 'recall', 'F1', 'AUC', 'accuracy'])
    new_pvalue_writer.writeheader()
    oo = {'epoch-new-pvalue':'cnn3', 'precision':'cnn3', 'recall': 'cnn3', 'F1': 'cnn3','AUC': 'cnn3', 'accuracy':'cnn3'}
    new_pvalue_writer.writerow(oo)
    
    
    f_new_maxsep = open('./results_esm2/cnn3_new_maxsep.csv', mode='a', encoding='utf-8', newline='')
    new_maxsep_writer= csv.DictWriter(f_new_maxsep, fieldnames=['epoch-new-maxsep', 'precision', 'recall', 'F1', 'AUC', 'accuracy'])
    new_maxsep_writer.writeheader()
    oo = {'epoch-new-maxsep':'cnn3', 'precision':'cnn3', 'recall': 'cnn3', 'F1': 'cnn3','AUC': 'cnn3', 'accuracy':'cnn3'}
    new_maxsep_writer.writerow(oo)
    
    
    
    f_price_pvalue = open('./results_esm2/cnn3_price_pvalue.csv', mode='a', encoding='utf-8', newline='')
    price_pvalue_writer= csv.DictWriter(f_price_pvalue, fieldnames=['epoch-price-pvalue', 'precision', 'recall', 'F1', 'AUC', 'accuracy'])
    price_pvalue_writer.writeheader()
    oo = {'epoch-price-pvalue':'cnn3', 'precision':'cnn3', 'recall': 'cnn3', 'F1': 'cnn3','AUC': 'cnn3', 'accuracy':'cnn3'}
    price_pvalue_writer.writerow(oo)
    
    
    
    f_price_maxsep = open('./results_esm2/cnn3_price_maxsep.csv', mode='a', encoding='utf-8', newline='')
    price_maxsep_writer= csv.DictWriter(f_price_maxsep, fieldnames=['epoch-price-maxsep', 'precision', 'recall', 'F1', 'AUC', 'accuracy'])
    price_maxsep_writer.writeheader()
    oo = {'epoch-price-maxsep':'cnn3', 'precision':'cnn3', 'recall': 'cnn3', 'F1': 'cnn3','AUC': 'cnn3', 'accuracy':'cnn3'}
    price_maxsep_writer.writerow(oo)
   
    
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

def main():
    #gc.collect()
    #torch.cuda.empty_cache()
    createCSV()
    seed_everything()
    ensure_dirs('./data/model')
    args = parse()
    torch.backends.cudnn.benchmark = True
    id_ec, ec_id_dict = get_ec_id_dict('./data/' + args.training_data + '.csv')
    ec_id = {key: list(ec_id_dict[key]) for key in ec_id_dict.keys()}
    #======================== override args ====================#
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    lr, epochs = args.learning_rate, args.epoch
    model_name = args.model_name
    print('==> device used:', device, '| dtype used: ',
          dtype, "\n==> args:", args)
    #======================== ESM embedding  ===================#
    # loading ESM embedding for dist map
 
    esm_emb = pickle.load(
        open('./data/distance_map/' + args.training_data + '_esm.pkl',
                'rb')).to(device=device, dtype=dtype)
                
    print(esm_emb)
    print(esm_emb.shape)
    dist_map = pickle.load(open('./data/distance_map/' + \
        args.training_data + '.pkl', 'rb')) 
    #print(dist_map)
    #print(dist_map.shape)
    #======================== initialize model =================#
    model = CNN3(args.hidden_dim, args.out_dim, device, dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.TripletMarginLoss(margin=1, reduction='mean')
    best_loss = float('inf')
    train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
    
    
    print("The number of unique EC numbers: ", len(dist_map.keys()))
    #======================== training =======-=================#
    # training             
    for epoch in range(1, epochs+1):
        epoch_start_time = time.time()
        train_loss = train(model, args, epoch, train_loader,
                           optimizer, device, dtype, criterion)
        
        elapsed = time.time() - epoch_start_time
        print('-' * 75)
        print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
              f'training loss {train_loss:6.10f}')
        print('-' * 75)
        
        #save a model every 1000 epoch after 4000 epoch
        if epoch % 1000 == 0 and epoch > 0:
            torch.save(model.state_dict(), './data/model/' + model_name + '_' + str(epoch) + '.pth')
            inferenceWirte(epoch, model)
            
        if epoch % args.adaptive_rate == 0 and epoch != epochs:
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, betas=(0.9, 0.999))
            
            # sample new distance map
            dist_map = get_dist_map(ec_id_dict, esm_emb, device, dtype, model=model)
            train_loader = get_dataloader(dist_map, id_ec, ec_id, args)
   

if __name__ == '__main__':
    main()
