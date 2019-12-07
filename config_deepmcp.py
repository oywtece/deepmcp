'''
config file
'''
# first dataset
n_one_hot_slot = 25 # num of one-hot slots in the 1st dataset
n_mul_hot_slot = 2 # num of mul-hot slots in the 1st dataset
max_len_per_slot = 5 # max num of fts per mul-hot slot in the 1st dataset
n_ft = 42301586 # num of unique fts in the 1st dataset
num_csv_col = 561 # num of cols in the csv file (1st dataset)
# total_n_slot = n_one_hot_slot + n_mul_hot_slot = 25+2 = 27
# the following indices are w.r.t. these total_n_slot(=27) slots, starting from slot idx 0
# in the sample csv data, slot idx 0 is bias; it does not belong to user or ad fts
user_ft_idx = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 25] # idx of user (& query) fts
ad_ft_idx = [1, 2, 19, 20, 21, 22, 23, 24, 26] # idx of ad fts

pre = './data/'
suf = '.csv'
train_file_name = [pre+'day_1'+suf, pre+'day_2'+suf] # can contain multiple file names
val_file_name = [pre+'day_3'+suf] # should contain only 1 file name
test_file_name = [pre+'day_4'+suf] # should contain only 1 file name

time_style = '%Y-%m-%d %H:%M:%S'
output_file_name = '0311_1430' # part of file and folder names for recording the output model and result
k = 10 # embedding dim for each ft
alpha = 5 # balancing para for the matching subnet
beta = 0.01 # balancing para for the correlation subnet
batch_size = 128 # batch size of the 1st dataset
kp_prob = 1.0 # keep prob in dropout; set to 1.0 if n_epoch = 1
opt_alg = 'Adagrad' # 'Adam'
eta = 0.05 # learning rate
max_num_lower_ct = 100 # early stop if the metric does not improve over the validation set after max_num_lower_ct times
n_epoch = 1 # number of times to loop over the 1st dataset
record_step_size = 200 # record auc and loss on the validation set after record_step_size times of mini_batch
layer_dim = [512, 256, 1] # prediction subnet FC layer dims, the last is the output layer, must be included
layer_dim_match = [512, 256] # matching subnet FC layer dims

# second dataset
train_file_name_corr = ['./data/corr.csv']
batch_size_corr = 128 # batch size of the 2nd dataset
layer_dim_corr = [512, 256] # correlation subnet FC layer dims
n_neg_used_corr = 4 # num of neg ads used for each target ad in the 2nd dataset
n_one_hot_slot_corr = 10 # num of one-hot slots per ad in the 2nd dataset
n_mul_hot_slot_corr = 2 # num of mul-hot slots per ad in the 2nd dataset
max_len_per_slot_corr = 10 # max num of fts per mul-hot slot in the 2nd dataset
num_csv_col_corr = 180 # num of cols in the csv file (2nd dataset)
n_epoch_corr = 2 # number of times to loop over the 2nd dataset
