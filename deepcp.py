# DeepCP - Deep Correlation and Prediction model

import numpy as np
import tensorflow as tf
import datetime
import ctr_funcs as func
import config_deepmcp as cfg
import os
import shutil

# config
str_txt = cfg.output_file_name
base_path = './tmp'
model_saving_addr = base_path + '/deepcp_' + str_txt + '/'
output_file_name = base_path + '/deepcp_' + str_txt + '.txt'
num_csv_col = cfg.num_csv_col
train_file_name = cfg.train_file_name
val_file_name = cfg.val_file_name
test_file_name = cfg.test_file_name
batch_size = cfg.batch_size
n_ft = cfg.n_ft
k = cfg.k
n_epoch = cfg.n_epoch
max_num_lower_ct = cfg.max_num_lower_ct
record_step_size = cfg.record_step_size
layer_dim = cfg.layer_dim
layer_dim_match = cfg.layer_dim_match
eta = cfg.eta # learning rate
opt_alg = cfg.opt_alg
n_one_hot_slot = cfg.n_one_hot_slot
n_mul_hot_slot = cfg.n_mul_hot_slot
max_len_per_slot = cfg.max_len_per_slot
beta = cfg.beta # for correlation loss
label_col_idx = 0
record_defaults = [[0]]*num_csv_col
record_defaults[0] = [0.0]
total_num_ft_col = num_csv_col - 1

## corr dataset - no test data for this dataset
train_file_name_corr = cfg.train_file_name_corr
batch_size_corr = cfg.batch_size_corr
layer_dim_corr = cfg.layer_dim_corr
n_one_hot_slot_corr = cfg.n_one_hot_slot_corr
n_mul_hot_slot_corr = cfg.n_mul_hot_slot_corr
max_len_per_slot_corr = cfg.max_len_per_slot_corr
n_epoch_corr = cfg.n_epoch_corr
n_neg_used_corr = cfg.n_neg_used_corr
# no label
num_csv_col_corr = cfg.num_csv_col_corr
record_defaults_corr = [[0]]*num_csv_col_corr
total_num_ft_col_corr = num_csv_col_corr
    
# create dir
if not os.path.exists(base_path):
    os.mkdir(base_path)

# remove dir
if os.path.isdir(model_saving_addr):
    shutil.rmtree(model_saving_addr)

# for DNN
idx_1 = n_one_hot_slot
idx_2 = idx_1 + n_mul_hot_slot*max_len_per_slot

###########################################################
###########################################################
print('Loading data start!')
tf.set_random_seed(123)

# load training data
train_ft, train_label = func.tf_input_pipeline(train_file_name, batch_size, n_epoch, label_col_idx, record_defaults)

n_val_inst = func.count_lines(val_file_name[0])
val_ft, val_label = func.tf_input_pipeline(val_file_name, n_val_inst, 1, label_col_idx, record_defaults)
n_val_batch = n_val_inst//batch_size

# load test data
test_ft, test_label = func.tf_input_pipeline_test(test_file_name, batch_size, 1, label_col_idx, record_defaults)
print('Loading data set 1 done!')

# load training data
train_ft_corr = func.tf_input_pipeline_wo_label(train_file_name_corr, batch_size_corr, n_epoch_corr, record_defaults_corr)
print('Loading data set 2 done!')

########################################################################
# partition input for correlation loss 
def partition_input_corr(x_input_corr):
    # generate idx_list
    len_list = []
    
    # 2 - tar & ctxt
    for i in range(n_neg_used_corr+2):
        len_list.append(n_one_hot_slot_corr)
        len_list.append(n_mul_hot_slot_corr*max_len_per_slot_corr)
    
    len_list = np.array(len_list)
    idx_list = np.cumsum(len_list)
            
    x_tar_one_hot_corr = x_input_corr[:, 0:idx_list[0]]
    x_tar_mul_hot_corr = x_input_corr[:, idx_list[0]:idx_list[1]]
    # shape=[None, n_mul_hot_slot, max_len_per_slot]
    x_tar_mul_hot_corr = tf.reshape(x_tar_mul_hot_corr, (-1, n_mul_hot_slot_corr, max_len_per_slot_corr))
            
    x_input_one_hot_dict_corr = {}
    x_input_mul_hot_dict_corr = {}
    
    for i in range(n_neg_used_corr+1):
        x_input_one_hot_dict_corr[i] = x_input_corr[:, idx_list[2*i+1]:idx_list[2*i+2]]
        temp = x_input_corr[:, idx_list[2*i+2]:idx_list[2*i+3]]
        x_input_mul_hot_dict_corr[i] = tf.reshape(temp, (-1, n_mul_hot_slot_corr, max_len_per_slot_corr))
    
    return x_tar_one_hot_corr, x_tar_mul_hot_corr, x_input_one_hot_dict_corr, x_input_mul_hot_dict_corr

# add mask
def get_masked_one_hot(x_input_one_hot):
    data_mask = tf.cast(tf.greater(x_input_one_hot, 0), tf.float32)
    data_mask = tf.expand_dims(data_mask, axis = 2)
    data_mask = tf.tile(data_mask, (1,1,k))
    # output: (?, n_one_hot_slot, k)
    data_embed_one_hot = tf.nn.embedding_lookup(emb_mat, x_input_one_hot)
    data_embed_one_hot_masked = tf.multiply(data_embed_one_hot, data_mask)
    return data_embed_one_hot_masked

def get_masked_mul_hot(x_input_mul_hot):
    data_mask = tf.cast(tf.greater(x_input_mul_hot, 0), tf.float32)
    data_mask = tf.expand_dims(data_mask, axis = 3)
    data_mask = tf.tile(data_mask, (1,1,1,k))
    # output: (?, n_mul_hot_slot, max_len_per_slot, k)
    data_embed_mul_hot = tf.nn.embedding_lookup(emb_mat, x_input_mul_hot)
    data_embed_mul_hot_masked = tf.multiply(data_embed_mul_hot, data_mask)
    # output: (?, n_mul_hot_slot, k)
    data_embed_mul_hot_masked = tf.reduce_sum(data_embed_mul_hot_masked, 2)
    return data_embed_mul_hot_masked

# output: (?, n_one_hot_slot + n_mul_hot_slot, k)
def get_concate_embed(x_input_one_hot, x_input_mul_hot):
    data_embed_one_hot = get_masked_one_hot(x_input_one_hot)
    data_embed_mul_hot = get_masked_mul_hot(x_input_mul_hot)
    data_embed_concat = tf.concat([data_embed_one_hot, data_embed_mul_hot], 1)
    return data_embed_concat

# input: (?, n_slot*k)
# output: (?, 1)
def get_pred_output(data_embed_concat):
    # include output layer
    n_layer = len(layer_dim)
    data_embed_dnn = tf.reshape(data_embed_concat, [-1, (n_one_hot_slot + n_mul_hot_slot)*k])
    cur_layer = data_embed_dnn
    # loop to create DNN struct
    for i in range(0, n_layer):
        # output layer, linear activation
        if i == n_layer - 1:
            cur_layer = tf.matmul(cur_layer, weight_dict[i]) + bias_dict[i]
        else:
            cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight_dict[i]) + bias_dict[i])
            cur_layer = tf.nn.dropout(cur_layer, keep_prob)
    
    y_hat = cur_layer
    return y_hat

# correlation loss input
def get_corr_output(x_input_corr):
    x_tar_one_hot_corr, x_tar_mul_hot_corr, x_input_one_hot_dict_corr, x_input_mul_hot_dict_corr = \
        partition_input_corr(x_input_corr)
    
    data_embed_tar = get_concate_embed(x_tar_one_hot_corr, x_tar_mul_hot_corr)
    data_vec_tar = tf.reshape(data_embed_tar, [-1, (n_one_hot_slot_corr + n_mul_hot_slot_corr)*k])
    
    n_layer_corr = len(layer_dim_corr)
    cur_layer = data_vec_tar
    for i in range(0, n_layer_corr):
        if i == n_layer_corr - 1:
            cur_layer = tf.nn.tanh(tf.matmul(cur_layer, weight_dict_corr[i]) + bias_dict_corr[i])
        else:
            cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight_dict_corr[i]) + bias_dict_corr[i])
    data_rep_tar = cur_layer
    
    # idx 0 - pos, idx 1 -- neg
    inner_prod_dict = {}
    for mm in range(n_neg_used_corr + 1):
        cur_data_embed = get_concate_embed(x_input_one_hot_dict_corr[mm], \
                                           x_input_mul_hot_dict_corr[mm])
        cur_data_vec = tf.reshape(cur_data_embed, [-1, (n_one_hot_slot_corr + n_mul_hot_slot_corr)*k])
        cur_layer = cur_data_vec
        for i in range(0, n_layer_corr):
            if i == n_layer_corr - 1:
                cur_layer = tf.nn.tanh(tf.matmul(cur_layer, weight_dict_corr[i]) + bias_dict_corr[i])
            else:
                cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight_dict_corr[i]) + bias_dict_corr[i])
        cur_data_rep = cur_layer
        # each ele - None*1
        inner_prod_dict[mm] = tf.reduce_sum(tf.multiply(data_rep_tar, cur_data_rep), 1, \
                            keep_dims=True)
    
    return inner_prod_dict

###########################################################
###########################################################
# input for l1 - prediction loss
x_input = tf.placeholder(tf.int32, shape=[None, total_num_ft_col])
# shape=[None, n_one_hot_slot]
x_input_one_hot = x_input[:, 0:idx_1]
x_input_mul_hot = x_input[:, idx_1:idx_2]
# shape=[None, n_mul_hot_slot, max_len_per_slot]
x_input_mul_hot = tf.reshape(x_input_mul_hot, (-1, n_mul_hot_slot, max_len_per_slot))

# input for corr loss
x_input_corr = tf.placeholder(tf.int32, shape=[None, total_num_ft_col_corr])

# target vec for l1
y_target = tf.placeholder(tf.float32, shape=[None, 1])

# dropout keep prob
keep_prob = tf.placeholder(tf.float32)
# emb_mat dim add 1 -> for padding (idx = 0)
with tf.device('/cpu:0'):
    emb_mat = tf.Variable(tf.random_normal([n_ft + 1, k], stddev=0.01))

################################
# prediction subnet FC layers, including output layer
n_layer = len(layer_dim)
in_dim = (n_one_hot_slot + n_mul_hot_slot)*k
weight_dict = {}
bias_dict = {}

# loop to create DNN vars
for i in range(0, n_layer):
    out_dim = layer_dim[i]
    weight_dict[i] = tf.Variable(tf.random_normal(shape=[in_dim, out_dim], stddev=np.sqrt(2.0/(in_dim+out_dim))))
    bias_dict[i] = tf.Variable(tf.constant(0.1, shape=[out_dim]))
    in_dim = layer_dim[i]

################################
# correlation subnet FC layers
n_layer_corr = len(layer_dim_corr)
in_dim_corr = (n_one_hot_slot_corr + n_mul_hot_slot_corr)*k
weight_dict_corr = {}
bias_dict_corr = {}

for i in range(0, n_layer_corr):
    out_dim_corr = layer_dim_corr[i]
    weight_dict_corr[i] = tf.Variable(tf.random_normal(shape=[in_dim_corr, out_dim_corr],\
                        stddev=np.sqrt(2.0/(in_dim_corr+out_dim_corr))))
    bias_dict_corr[i] = tf.Variable(tf.constant(0.1, shape=[out_dim_corr]))
    in_dim_corr = layer_dim_corr[i]
################################

data_embed_concat = get_concate_embed(x_input_one_hot, x_input_mul_hot)
y_hat = get_pred_output(data_embed_concat)
inner_prod_dict_corr = get_corr_output(x_input_corr)

loss_ctr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=y_target))
# logloss
y_corr_cast_1 = tf.ones_like(inner_prod_dict_corr[0])
y_corr_cast_0 = tf.zeros_like(inner_prod_dict_corr[0])
# pos
loss_corr = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=inner_prod_dict_corr[0], \
    labels=y_corr_cast_1))
# neg
for i in range(n_neg_used_corr):
    loss_corr += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=inner_prod_dict_corr[i+1], \
                 labels=y_corr_cast_0))

loss = loss_ctr + beta*loss_corr

#############################
# prediction
#############################
pred_score = tf.sigmoid(y_hat)

if opt_alg == 'Adam':
    optimizer = tf.train.AdamOptimizer(eta).minimize(loss)
else:
    # default
    optimizer = tf.train.AdagradOptimizer(eta).minimize(loss)

########################################
# Launch the graph.
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.3

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess, coord)

    saver_val = tf.train.Saver()
    train_loss_list = []
    val_auc_list = []
    best_n_round = 0
    best_val_auc = 0            
    lower_ct = 0
    early_stop_flag = 0

    val_ft_inst, val_label_inst = sess.run([val_ft, val_label])

    func.print_time()
    print('Start train loop')
    
    epoch = -1
    try:
        while not coord.should_stop():           
            epoch += 1  
            train_ft_inst, train_label_inst = sess.run([train_ft, train_label])
            train_label_inst = np.transpose([train_label_inst])            
            
            train_ft_corr_inst = sess.run(train_ft_corr)
            
            # training
            sess.run(optimizer, feed_dict={x_input:train_ft_inst, y_target:train_label_inst, \
                                           x_input_corr:train_ft_corr_inst, keep_prob:1.0})
            
            # record loss and accuracy every step_size generations
            if (epoch+1)%record_step_size == 0:                
                train_loss_temp = sess.run(loss, feed_dict={ \
                                           x_input:train_ft_inst, y_target:train_label_inst, \
                                           x_input_corr:train_ft_corr_inst, keep_prob:1.0})
                train_loss_list.append(train_loss_temp)                
 
                val_pred_score_all = []
                val_label_all = []
                
                for iii in range(n_val_batch):
                    # get batch
                    start_idx = iii*batch_size
                    end_idx = (iii+1)*batch_size
                    cur_val_ft = val_ft_inst[start_idx: end_idx]
                    cur_val_label = val_label_inst[start_idx: end_idx]
                    # pred score
                    cur_val_pred_score = sess.run(pred_score, feed_dict={ \
                                            x_input:cur_val_ft, keep_prob:1.0})
                    val_pred_score_all.append(cur_val_pred_score.flatten())
                    val_label_all.append(cur_val_label)   
                    
                # calculate auc
                val_pred_score_re = func.list_flatten(val_pred_score_all)
                val_label_re = func.list_flatten(val_label_all)
                val_auc_temp, _, _ = func.cal_auc(val_pred_score_re, val_label_re)
                # record all val results    
                val_auc_list.append(val_auc_temp)
                 
                # record best and save models
                if val_auc_temp > best_val_auc:
                    best_val_auc = val_auc_temp
                    best_n_round = epoch
                    # Save the variables to disk
                    save_path = saver_val.save(sess, model_saving_addr)
                    print("Model saved in: %s" % save_path)
                # count of consecutive lower
                if val_auc_temp < best_val_auc:
                     lower_ct += 1
                # once higher or equal, set to 0
                else:
                     lower_ct = 0
                
                if lower_ct >= max_num_lower_ct:
                    early_stop_flag = 1
                
                auc_and_loss = [epoch+1, train_loss_temp, val_auc_temp]
                # round to given number of decimals
                auc_and_loss = [np.round(xx,4) for xx in auc_and_loss]
                func.print_time() 
                print('Generation # {}. Train Loss: {:.4f}. Val Avg AUC: {:.4f}.'\
                      .format(*auc_and_loss))

            # stop while loop    
            if early_stop_flag == 1:
                break 
                
    except tf.errors.OutOfRangeError:
        func.print_time()
        print('Done training -- epoch limit reached')
    
    # restore model
    saver_val.restore(sess, model_saving_addr)
    print("Model restored.")
            
    # load test data
    test_pred_score_all = []
    test_label_all = []
    test_loss_all = []
    try:
        while True:
            test_ft_inst, test_label_inst = sess.run([test_ft, test_label])
            cur_test_pred_score = sess.run(pred_score, feed_dict={ \
                                    x_input:test_ft_inst, keep_prob:1.0})
            test_pred_score_all.append(cur_test_pred_score.flatten())
            test_label_all.append(test_label_inst)
            
            cur_test_loss = sess.run(loss_ctr, feed_dict={ \
                                    x_input:test_ft_inst, \
                                    y_target: np.transpose([test_label_inst]), keep_prob:1.0})
            test_loss_all.append(cur_test_loss)

    except tf.errors.OutOfRangeError:
        func.print_time()
        print('Done testing -- epoch limit reached')    
    finally:
        coord.request_stop()
        
    coord.join(threads) 
         
    # calculate auc
    test_pred_score_re = func.list_flatten(test_pred_score_all)
    test_label_re = func.list_flatten(test_label_all)
    test_auc, _, _ = func.cal_auc(test_pred_score_re, test_label_re)
    test_rmse = func.cal_rmse(test_pred_score_re, test_label_re)
    test_loss = np.mean(test_loss_all)
    
    # rounding
    test_auc = np.round(test_auc, 4)
    test_rmse = np.round(test_rmse, 4)
    test_loss = np.round(test_loss, 5)
    train_loss_list = [np.round(xx,4) for xx in train_loss_list]
    val_auc_list = [np.round(xx,4) for xx in val_auc_list]
    
    print('test_auc = ', test_auc)
    print('test_rmse =', test_rmse)
    print('test_loss =', test_loss)
    print('train_loss_list =', train_loss_list)
    print('val_auc_list =', val_auc_list)
    
    # write output to file
    with open(output_file_name, 'a') as f:
        now = datetime.datetime.now()
        time_str = now.strftime(cfg.time_style)
        f.write(time_str + '\n')
        f.write('train_file_name = ' + train_file_name[0] + '\n')
        f.write('learning_rate = ' + str(eta) \
                + ', beta = ' + str(beta) \
                + ', n_epoch = ' + str(n_epoch) \
                + ', emb_dize = ' + str(k) + '\n')
        f.write('test_auc = ' + str(test_auc) + '\n')
        f.write('test_rmse = ' + str(test_rmse) + '\n')
        f.write('test_loss = ' + str(test_loss) + '\n')
        f.write('train_loss_list =' + str(train_loss_list) + '\n')
        f.write('val_auc_list =' + str(val_auc_list) + '\n')
        f.write('-'*50 + '\n')
