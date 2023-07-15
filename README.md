# PCCLM: A parallel convolutional contrastive learning model for enzyme function prediction
In our study, we propose a parallel convolutional contrastive learning model called PCCLM for predicting the EC number of enzymes.
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

1.Before training, we need to preprocess the training and test sets: <br>
 python pre-train.py <br>

 pre-train.py:<br>
  csv_to_fasta("data/split100.csv", "data/split100.fasta") <br>
  retrive_esm2_embedding(train_file) <br>
  train_fasta_file = mutate_single_seq_ECs_forESM2(train_file) <br>
  retrive_esm2_embedding(train_fasta_file) <br>
  compute_esm2_distance(train_file) <br>

  test_data_price = "price" <br>
  test_data_new = "new" <br>
  retrive_esm2_embedding(test_data_new) <br>
  retrive_esm2_embedding(test_data_price) <br>
  
2. Please create a folder "results_esm2"

3.Train the model. It is automatically tested every 1000 epochs. <br>

python ./train-triplet-cnn3.py --training_data split100 --model_name split100_triplet_cnn3 --epoch 15000 <br>
# Contact:
Xindi Yu: yuxindi53@foxmail.com
