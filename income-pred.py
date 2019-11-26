import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import lightgbm as lgb


# reading data and handling unknowns
def openAndHandleUnknowns(fileName):
    return pd.read_csv(fileName, na_values={
        'Year of Record': [0, '#N/A', 'na', 'nA', 'NA', 'unknown', '#NUM!', ''],
        'Gender': [0, '#N/A', 'na', 'nA', 'NA', 'unknown'],
        'Age': [0, '#N/A', 'na', 'nA', 'NA', 'unknown'],
        'Country': ['#N/A', 'na', 'nA', 'NA', 'unknown', '0'],
        'Size of City': ['#N/A', 'na', 'nA', 'NA', 'unknown'],
        'Profession': ['#N/A', 'na', 'nA', 'NA', 'unknown'],
        'University Degree': [0, '#N/A', 'na', 'nA', 'NA', 'unknown'],
        'Wears Glasses': ['#N/A', 'na', 'nA', 'NA', 'unknown'],
        'Hair Color': [0, '#N/A', 'na', 'nA', 'NA', 'unknown'],
        'Body Height [cm]': ['#N/A', 'na', 'nA', 'NA', 'unknown', 0],

        'Housing Situation': [0, '#N/A', 'na', 'nA', 'NA', 'unknown'],
        'Crime Level in the City of Employement': ['na', 'nA', '#N/A', 'NA'],
        'Work Experience in Current Job [years]': ['#NUM!', 'na', '#N/A', 'NA', 'nA'],
        'Satisfation with employer': ['#N/A', 'na', 'nA', 'NA', 'unknown'],
        'Yearly Income in addition to Salary (e.g. Rental Income)': [],
        'Total Yearly Income [EUR]': []
    }, low_memory=False)


# handling NaN
def dfFillNaN(data):
    data['Age'].fillna(data['Age'].mean(), inplace=True)
    data['Crime Level in the City of Employement'].fillna(data['Crime Level in the City of Employement'].mean(), inplace=True)
    data['Work Experience in Current Job [years]'].fillna(data['Work Experience in Current Job [years]'].mean(), inplace=True)
    data['Year of Record'].fillna(data['Year of Record'].mean(), inplace=True)

    data['Yearly Income in addition to Salary (e.g. Rental Income)'] = \
        data['Yearly Income in addition to Salary (e.g. Rental Income)'].str.split(' ').str[0]. \
            str.strip().astype('float64')
    data['Gender'].replace(to_replace='f', value='female', inplace=True)
    return data


# encoding the dataset using target encoding
def target_encode(train_df, target_variable, cat_cols, alpha):
    te_train = train_df.copy()
    globalmean = train_df[target_variable].mean()
    cat_map = dict()
    def_map = dict()

    for col in cat_cols:
        cat_count = train_df.groupby(col).size()
        target_cat_mean = train_df.groupby(col)[target_variable].mean()
        reg_smooth_val = ((target_cat_mean * cat_count) + (globalmean * alpha)) / (cat_count + alpha)

        te_train.loc[:, col] = te_train[col].map(reg_smooth_val)
        te_train[col].fillna(globalmean, inplace=True)

        cat_map[col] = reg_smooth_val
        def_map[col] = globalmean
    return te_train, cat_map, def_map


print('loading data...')
df = openAndHandleUnknowns('tcd-ml-1920-group-income-train.csv')
sub_df = openAndHandleUnknowns('tcd-ml-1920-group-income-test.csv')

print('removing duplicates...')
df.drop_duplicates(inplace=True)

print('filling NaN values...')
df = dfFillNaN(df)
sub_df = dfFillNaN(sub_df)

# applying log transform on y
df['Total Yearly Income [EUR]'] = df['Total Yearly Income [EUR]']

y = df['Total Yearly Income [EUR]']
instance = pd.DataFrame(sub_df['Instance'], columns=['Instance'])

# features being considered for prediction
features = ['Year of Record', 'Housing Situation', 'Crime Level in the City of Employement',
            'Work Experience in Current Job [years]', 'Satisfation with employer',
            'Gender', 'Age', 'University Degree', 'Country', 'Size of City', 'Profession',
            # 'Wears Glasses', 'Hair Color', 'Body Height [cm]',
            'Yearly Income in addition to Salary (e.g. Rental Income)']

categorical_columns = ['Housing Situation', 'Satisfation with employer', 'Gender', 'Country', 'Profession',
                       'University Degree']

df = df[features + ['Total Yearly Income [EUR]']]
sub_df = sub_df[features]

# Feature modifications
# Target Encoding
df, target_mapping, default_mapping = target_encode(df, 'Total Yearly Income [EUR]', categorical_columns, 10)
for column in categorical_columns:
    sub_df.loc[:, column] = sub_df[column].map(target_mapping[column])
    sub_df[column].fillna(default_mapping[column], inplace=True)


# removing income column
df = df[features]
print('Training Model!!')
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

params = {}
params['learning_rate'] = 0.001
params['boosting_type'] = 'gbdt'
params['metric'] = 'mae'
params['max_depth'] = 30
params['verbosity'] = -1
params['objective'] = 'tweedie'
params['num_threads'] = 4
params['feature_fraction'] = 0.8

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)
model = lgb.train(params=params, train_set=train_data, num_boost_round=100000, valid_sets=[train_data, test_data],
                  verbose_eval=1000, early_stopping_rounds=500)

print('predicting Y...')
y_pred = model.predict(X_test)

print("MAE: %.2f" % mean_absolute_error(y_test, y_pred))
print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

##################################################################################################################
print('\nPredicting the output!')
y_sub = model.predict(sub_df)

print('creating final csv...')
income = pd.DataFrame(y_sub, columns=['Total Yearly Income [EUR]'])
ans = instance.join(income)

ans.to_csv('kaggle-output.csv', index=False)
