# tcdml1920: Income Prediction Competition - Team 14
Machine Learning Group Competition code as part of the 2019/20 Machine Learning module at Trinity College Dublin.

#### Competition link -> [tcd-ml-comp-201920-income-pred-group](https://www.kaggle.com/c/tcd-ml-comp-201920-income-pred-group)

Team14:  
[Mukesh Arambakam](https://github.com/mukeshmk) - 19301497  
[Ellen Mullooly](https://github.com/ellenmullooly) - 15320582  
[Alex Fields](https://github.com/fieldsal) - 15314665  

Please check [tcdml1920-income-ind](https://github.com/mukeshmk/tcdml1920-income-ind) repo for the base of the source code used here.

#### algorithm used [light-gbm](https://lightgbm.readthedocs.io/en/latest/index.html)
parameters:

```python
params['learning_rate'] = 0.001
params['boosting_type'] = 'gbdt'
params['metric'] = 'mae'
params['max_depth'] = 30
params['verbosity'] = -1
params['objective'] = 'tweedie'
params['num_threads'] = 4
params['feature_fraction'] = 0.8
```
for detailed description of the parameters click [here](https://lightgbm.readthedocs.io/en/latest/Parameters.html).

#### Leaderboard Score
Private - 10269.49072  
Public - 10450.95466
