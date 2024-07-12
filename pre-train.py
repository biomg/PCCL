from CLEAN.utils import *

train_file = "split100"

#esm1b
csv_to_fasta("data/split100.csv", "data/split100.fasta")
retrive_esm1b_embedding("split100")
train_fasta_file = mutate_single_seq_ECs(train_file)
retrive_esm1b_embedding(train_fasta_file)
compute_esm_distance(train_file)

test_data_price = "price"
test_data_new = "new"
retrive_esm1b_embedding(test_data_new)
retrive_esm1b_embedding(test_data_price)

#esm2
retrive_esm2_embedding(train_file)
train_fasta_file = mutate_single_seq_ECs_forESM2(train_file)
retrive_esm2_embedding(train_fasta_file)
compute_esm2_distance(train_file)

test_data_price = "price"
test_data_new = "new"
retrive_esm2_embedding(test_data_new)
retrive_esm2_embedding(test_data_price)
