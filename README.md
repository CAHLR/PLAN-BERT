# PLAN-BERT

Keras implementation of PLAN-BERT: A Transformer-based sequential basket recommender algorithm that incorporates future item information.

```
    pip install PLANBERT
```

Adapted from Shao, Guo, & Pardos ([AAAI 2021](https://ojs.aaai.org/index.php/AAAI/article/view/17751)) for a manuscript submitted to a special issue of TKDE (under review). Scripts by Erzhuo Shao.

## Requirements
Keras>=2.3.1
numpy
pandas
tensorflow-gpu
tqdm



## Installing PLAN-BERT ##

You can simply run:
```
    pip install PLANBERT
``` 
Alternatively, if `pip` poses some problems, you can clone the repository as such and then run the `setup.py` script manually.

```
    git clone https://github.com/CAHLR/PLAN-BERT.git
    cd PLAN-BERT
    python setup.py install
```

# Preparing Data and Running Model #

The following serves as a mini-tutorial for how to get started with PLAN-BERT.
Prepare csv data




## Input Format ##

The accepted input formats are Pandas DataFrames, which much includes columns 'user', 't', 'item'. Columns whose name include 'feat' would be considered as features. Time slots are relative, which means the minimum time slots of any user are reocmmended to be 0. Since PLAN-BERT is a basket-level recommendation model, one time slot may contain multiple items. All following columns are undeerstood as features. User or item features are not distingushed in DataFrame, all features must be provided in each row. The names of all columns are arbitrary.

| 'user' | 't' | 'item' | 'feat_1'(Optional) | ... | 'feat_N'(Optional) |
|:------:|:---:|:------:|:--------:|:--------:|:---:|
| User1  |  0  | item0  |          |          |     |
| User1  |  0  | item1  |          |          |     |
| User1  |  1  | item2  |          |          |     |
| User1  |  2  | item1  |          |          |     |
| User1  |  3  | item3  |          |          |     |
| User2  |  0  | item0  |          |          |     |
| ...    |     |        |          |          |     |

The output of predict funtion includes 4 columns, 'user', 't', 'item', 'prob'. the number of future time slots is pred_time_slices, 't' start from the maximum historical time slots + 1. Each basket includes items_per_slice items, which are order by their predicted probability. 
The maximum predicted time slots + historical time slots can exceed the width of PLAnBERT. If the number of required future time slots (e.g., 3) + the length of history (e.g., 3) is greater than the width of PLAN-BERT (e.g., 5), several beginning time slots (3+3-5) would be droped out from input DataFrame. So 

| 'user' | 't' | 'item' | 'prob' |
|:------:|:---:|:------:|:------:|
| User1  |  3  | item0  |        |
| User1  |  3  | item1  |        |
| User1  |  4  | item2  |        |
| User1  |  5  | item1  |        |
| User1  |  5  | item3  |        |
| User2  |  3  | item0  |        |
| ...    |     |        |        |

## Creating and Training Models ##

The process of creating and training models in PLAN-BERT resemble that of SciKit Learn. 

```python
from PLANBERT.Model import PLANBERT
# Load the training, validation, and testing set.
import pandas as pd
master_csv = pd.read_csv('./example/example_master.csv')
train_len, train_valid_len = 153681, 153681+31661
train_csv, valid_csv, test_csv = master_csv.iloc[:train_len], master_csv.iloc[train_len:train_valid_len], master_csv.iloc[train_valid_len:]

# Train a PLAN-BERT with training set and validation set without checkpoint.

#planbert = PLANBERT(num_times=6, num_items=10000, feat_dims=[5000, 1000], cuda_num=0) # [ Number of time slots, Number of items, [Number of features], ID of GPU]
planbert = PLANBERT(master_csv) # Automatically extract network hyper-parameters from DataFrame.
#planbert.fit(train_csv, valid_csv)
$planbert.test(test_csv, h_list=[9], r_list=[3], pool_size=25)

# Obtain the output schedule. We note that the test_csv should only include historical items and future reference items. We should sample test_csv before feeding it into planbert.predict.
test_csv_history = test_csv[test_csv['t'] < 2]
test_csv_future = test_csv[test_csv['t'] >= 2]
predict = planbert.predict(
    test_csv_history, # Historical DataFrame
    test_csv_future, # Future DataFrame, whose columns 't' would be useless in 'wishlist' mode.
    mode='time', # PLAN-BERT's mode (time/wishlist)
    pred_time_slices=4, # The number of required time slots in the future schedule.
    items_per_slice=20 # The number of required items in each future time slots.
)
```
# Internal Data Format #

        | 0  0  0  0  1  0 |
        | 0  1  0  0  0  0 |
        | 0  0  0  0  0  1 |
        | 0  0  0  0  0  1 |
        | 0  0  1  0  0  0 |   

The above example shows the schedule of one user. It is of shape [5, 6], in which 5 is the number of time slots, 6 is the number of items. Similarly, the output of PLAN-BERT is of shape [ U, T, I ], where U is the number of users, T is the number of time slots, I is the number of items.
  
  
