# PCCL: Parallel convolutional contrastive learning method for enzyme function prediction
In our study, we propose a parallel convolutional contrastive learning method called PCCL for predicting the EC number of enzymes.
# Dependency:
python 3.10.4 <br>
pytorch 1.11.0 <br>
cudatoolkit 11.3 <br>
matplotlib 3.7.0 <br>
numpy 1.22.3 <br>
pandas 1.4.2 <br>
scikit_learn 1.2.0 <br>
scipy 1.7.3 <br>
tqdm 4.64.0 <br>
fair-esm 2.0.0 <br>
# Download the required ESM-2 source code：
git clone https://github.com/facebookresearch/esm.git

# Data：
Training set: split100.csv (including 227362 enzyme sequences) <br>
Test set:  <br>
new.csv (including 392 enzyme sequences) <br>
price.csv (including 149 enzyme sequences) <br>

# Supported GPUs:
Now it supports GPUs. The code support GPUs and CPUs, it automatically check whether you server install GPU or not, it will proritize using the GPUs if there exist GPUs.

# Training and testing:

1、Before training, we need to preprocess the training and test sets:

python pre-train.py

2、Please create a folder "results_esm2"

3、Train the model. It is automatically tested every 1000 epochs. 

python ./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_cnn3 --epoch 15000

4、If you want to reproduce the experiment in section 3.2.1, you need to do the following:

Directly execute "python./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_cnn3 --epoch 15000". The experimental results of cnn-3 in Figure 5 can be obtained.

Change “model = CNN3(args.hidden_dim, args.out_dim, device, dtype)” in train-triplet-cnn3.py to “model = CNN2(args.hidden_dim, args.out_dim, device, dtype)”. run "python ./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_cnn2 --epoch 15000". The experimental results of cnn-2 in Figure 5 can be obtained.

Change “model = CNN3(args.hidden_dim, args.out_dim, device, dtype)” in train-triplet-cnn3.py to “model = CNN4(args.hidden_dim, args.out_dim, device, dtype)”. run "python ./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_cnn4 --epoch 15000". The experimental results of cnn-4 in Figure 5 can be obtained.

5、If you want to reproduce the experiment in Section 3.2.2, you need to do the following:

Directly execute "python ./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_cnn3 --epoch 15000". Experimental results with the same convolution kernel size as shown in Figure 6 can be obtained.

Change the sizes of the three convolution kernels of "class CNN3" in model.py to 1, 3, and 5. Run "python ./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_cnn3 --epoch 15000". Experimental results of different convolution kernel sizes in Figure 6 can be obtained.

6、If you want to reproduce the experiment in section 3.2.3, you need to do the following:

Set drop_out of "class CNN3" in model.py to 0.05, 0.15, 0.25, and 0.35, respectively. Run "python ./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_cnn3 --epoch 15000", respectively. Experimental results of different loss rates in Figure 7 can be obtained.

7、If you want to reproduce the experiment in section 3.3, you need to do the following:

Directly execute "python ./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_cnn3 --epoch 15000". The experimental results of PCCL in Figure 8 can be obtained.

The results of the other methods in Figure 8 are from Yu et al.

8、If you want to reproduce the experiment in section 3.4, you need to do the following:

Directly execute "python ./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_cnn3 --epoch 15000". The experimental results of ESM-2+CNN (PCCL) in Figure 9 can be obtained.

Execute "python ./train-triplet.py --training_data split100 --model_name split100_triplet_cnn3_esm1b --epoch 15000". The experimental results of ESM-1b+CNN in Figure 9 can be obtained.

Change “model = CNN3(args.hidden_dim, args.out_dim, device, dtype)” in train-triplet-cnn3.py to “model = LayerNormNet(args.hidden_dim, args.out_dim, device, dtype)”. Run "python ./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_fcnn --epoch 15000". The experimental results of ESM-2+FCNN in Figure 9 can be obtained.

# Contact:
Xindi Yu: yuxindi53@foxmail.com
