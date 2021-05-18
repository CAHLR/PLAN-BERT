# PLAN-BERT

Keras implementation of the PLAN-BERT. A Transformer-based sequential baskets recommender system, which incorporates future items.

```
    pip install PLANBERT
```

Based on the paper http://askoski.berkeley.edu/~zp/papers/academic_plan_generation.pdf and TKDE paper. All scripts are written by Erzhuo Shao.

## Requirements
Keras==2.3.1
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




## Input and Output Data ##

The accepted input formats are Pandas DataFrames. The first three columns denotes the id of users, time slots, and items. Time slots are relative, which means the minimum time slots of any user are reocmmended to be 0. Since PLAN-BERT is a basket-level recommendation model, one time slot may contain multiple items. All following columns are undeerstood as features. User or item features are not distingushed in DataFrame, all features must be provided in each row. The names of all columns are arbitrary.

| 'user' | 't' | 'item' | 'feat_1' | 'feat_2' | ... |
|:------:|:---:|:------:|:--------:|:--------:|:---:|
| User1  | t1  | item1  |          |          |     |
| User1  | t1  | item2  |          |          |     |
| User1  | t2  | item3  |          |          |     |
| User1  | t2  | item2  |          |          |     |
| User1  | t3  | item4  |          |          |     |
| User2  | t1  | item1  |          |          |     |
| ...    |     |        |          |          |     |

## Creating and Training Models ##

The process of creating and training models in PLAN-BERT resemble that of SciKit Learn. 

```python
from PLANBERT.Model import PLANBERT
# Load the training, validation, and testing set.
import pandas as pd
train_csv = pd.read_csv('../example/example_train.csv')
valid_csv = pd.read_csv('../example/example_valid.csv')
test_csv = pd.read_csv('../example/example_train.csv')

# Train a PLAN-BERT with training set and validation set without checkpoint.
planbert = PLANBERT(0, 0, 0, 0, train_csv) # [ Number of time slots, Number of items, [Number of features], ID of GPU]
planbert.fit(train_csv, valid_csv)
planbert.test(test_csv, h_list=[9], r_list=[3], pool_size=25)

# Obtain the output schedule. We note that the test_csv should only include historical items and future reference items. We should sample test_csv before feeding it into planbert.predict.
history_dict = {iter:6 for iter in test_csv['user'].unique()[:10]}
predict = planbert.predict(test_csv, 'time', history_dict) # [Testing set, PLAN-BERT's mode ('time'/'wishlist'), Number of historical time slots]
```
# Internal Data Format #

        | 0  0  0  0  1  0 |
        | 0  1  0  0  0  0 |
        | 0  0  0  0  0  1 |
        | 0  0  0  0  0  1 |
        | 0  0  1  0  0  0 |   

The above example shows the schedule of one user. It is of shape [5, 6], in which 5 is the number of time slots, 6 is the number of items. Similarly, the output of PLAN-BERT is of shape [ U, T, I ], where U is the number of users, T is the number of time slots, I is the number of items.
  
  
