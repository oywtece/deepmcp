# Deep Matching, Correlation and Prediction (DeepMCP) Model

DeepMCP is a model for click-through rate (CTR) prediction. Most existing methods mainly model the feature-CTR relationship and suffer from the data sparsity issue. In contrast, DeepMCP models other types of relationships in order to learn more informative and statistically reliable feature representations, and in consequence to improve the performance of CTR prediction. In particular, DeepMCP contains three parts: a matching subnet, a correlation subnet and a prediction subnet. These subnets model the user-ad, ad-ad and feature-CTR relationship respectively. When these subnets are jointly optimized under the supervision of the target labels, the learned feature representations have both good prediction powers and good representation abilities. 

If you use this code, please cite the following paper:
* **Wentao Ouyang, Xiuwu Zhang, Shukui Ren, Chao Qi, Zhaojie Liu, Yanlong Du. Representation Learning-Assisted Click-Through Rate Prediction. In IJCAI, 2019.**

#### TensorFlow (TF) version
1.3.0

#### Abbreviation
ft - feature, slot == field

## Data Preparation (DeepMP)
Data is in the "csv" format, where each row contains an instance.\
Assume there are N unique fts. Fts need to be indexed from 1 to N. Use 0 for missing values or for padding.

We categorize fts as i) **one-hot** or **univalent** (e.g., user id, city) and ii) **mul-hot** or **multivalent** (e.g., words in ad title).

csv data format
* \<label\>\<one-hot fts\>\<mul-hot fts\>

We also need to define the max number of features per mul-hot ft slot (through the "max_len_per_slot" parameter) and perform trimming or padding accordingly. Please refer to the following example for more detail.

### Example
1. original fts (ft_name:ft_value)
* label:0, gender:male, age:27, query:apple, title:apple, title:fruit, title:fresh
* label:1, gender:female, age:35, query:shoes, query:winter, title:shoes, title:winter, title:warm, title:sales

2. csv fts (not converted to ft index yet)
* 0, male, 27, apple, 0, 0, apple, fruit, fresh
* 1, female, 35, shoes, winter, 0, shoes, winter, warm

#### Explanation
csv format settings:\
n_one_hot_slot = 2 # num of one-hot ft slots (gender, age)\
n_mul_hot_slot = 2 # num of mul-hot ft slots (query, title)\
max_len_per_slot = 3 # max num of fts per mul-hot ft slot

For the first instance, the mul-hot ft slot "query" contains only 1 ft "apple". We thus pad (max_len_per_slot - 1) zeros, resulting in "apple, 0, 0".\
For the second instance, the mul-hot ft slot "title" contains 4 fts. We thus only keep the first max_len_per_slot fts.

## Data Preparation (DeepCP/DeepMCP)
DeepCP/DeepMCP needs two datasets as input. Both are in the "csv" format.\
The first dataset is the same as that for DeepMP.\
The second dataset should contain a target ad, a context ad and N negative ads per row.

csv data format
* \<target one-hot fts\>\<target mul-hot fts\>\<ctxt one-hot fts\>\<ctxt mul-hot fts\>\<neg1 one-hot fts\>\<neg1 mul-hot fts\>...\<negN one-hot fts\>\<negN mul-hot fts\>

csv format settings:\
n_one_hot_slot_s = 2 # num of one-hot ft slots per ad in the second dataset\
n_mul_hot_slot_s = 2 # num of mul-hot ft slots per ad in the second dataset\
max_len_per_slot_s = 3 # max num of fts per mul-hot ft slot in the second dataset

## Source Code
1. **DeepMP** achieves the best tradeoff between prediction performance and model complexity. It needs only 1 dataset. (configs of the second dataset are useless) \[**_Recommended_**\]
2. DeepCP needs 2 datasets. Its performance is not as good as DeepMP.
3. DeepMCP also needs 2 datasets. It is the most complex and leads to the best performance.

* config_deepmcp.py -- config file
* ctr_funcs.py -- functions
* deepmp.py -- Deep Matching and Prediction (DeepMP) model
* deepcp.py -- Deep Correlation and Prediction (DeepCP) model
* deepmcp.py -- Deep Matching, Correlation and Prediction (DeepMCP) model

## Run the Code
First revise the config file, and then run the code
```bash
nohup python deepmp.py > [output_file_name] 2>&1 &
```
