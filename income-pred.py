import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing as pp
from sklearn.ensemble import RandomForestRegressor
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
    # TODO check why interpolate on 'Year of Record' is working?
    # df['Year of Record'] = np.floor(df['Year of Record'].interpolate(method='slinear'))
    data['Age'] = np.floor(data['Age'].interpolate(method='slinear'))
    data['Crime Level in the City of Employement'] = np.floor(
        data['Crime Level in the City of Employement'].interpolate(method='slinear'))
    data['Work Experience in Current Job [years]'] = np.floor(
        data['Work Experience in Current Job [years]'].interpolate(method='slinear'))
    # different methods of fillna for categorical values?
    data['Gender'].fillna('other', inplace=True)
    data['Profession'].fillna(method='ffill', inplace=True)
    data['University Degree'].fillna(method='ffill', inplace=True)
    data['Country'].fillna(method='ffill', inplace=True)
    data['Hair Color'].fillna(method='ffill', inplace=True)
    data['Housing Situation'].fillna(method='bfill', inplace=True)
    data['Housing Situation'].fillna(method='ffill', inplace=True)
    data['Satisfation with employer'].fillna(method='ffill', inplace=True)
    data['Year of Record'].fillna(method='ffill', inplace=True)

    data['Yearly Income in addition to Salary (e.g. Rental Income)'] = \
        data['Yearly Income in addition to Salary (e.g. Rental Income)'].str.split(' ').str[0]. \
            str.strip().astype('float64')
    data['Gender'].replace(to_replace='f', value='female', inplace=True)
    return data


def removeIncomeRows(data):
    outlierInc = detectOutlier(data['Total Yearly Income [EUR]'])
    data = data[~data["Total Yearly Income [EUR]"].isin(outlierInc)]

    data = data[(data['Total Yearly Income [EUR]'] >= 0)]
    return data


def detectOutlier(data):
    threshold = 3
    mean_1 = np.mean(data)
    std_1 = np.std(data)
    outliers = []
    for y in data:
        z_score = (y - mean_1) / std_1
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


# One Hot Encoding
def oheFeature(feature, encoder, data, df):
    ohedf = pd.DataFrame(data, columns=[feature + ': ' + str(i.strip('x0123_')) for i in encoder.get_feature_names()])
    ohedf.drop(ohedf.columns[len(ohedf.columns) - 1], axis=1, inplace=True)
    df = pd.concat([df, ohedf], axis=1)
    del df[feature]
    return df


def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))


def target_encode(trn_series=None, tst_series=None, target=None, min_samples_leaf=1, smoothing=1, noise_level=0):
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply aver
    assert len(trn_series) == len(target)
    assert trn_series.name == tst_series.name
    temp = pd.concat([trn_series, target], axis=1)
    # Compute target mean
    averages = temp.groupby(by=trn_series.name)[target.name].agg(["mean", "count"])
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages["count"] - min_samples_leaf) / smoothing))
    # Apply average function to all target data
    prior = target.mean()
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages["mean"] * smoothing
    averages.drop(["mean", "count"], axis=1, inplace=True)
    # Apply averages to trn and tst series
    ft_trn_series = pd.merge(
        trn_series.to_frame(trn_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=trn_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_trn_series.index = trn_series.index
    ft_tst_series = pd.merge(
        tst_series.to_frame(tst_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=tst_series.name,
        how='left')['average'].rename(trn_series.name + '_mean').fillna(prior)
    # pd.merge does not keep the index so restore it
    ft_tst_series.index = tst_series.index
    return add_noise(ft_trn_series, noise_level), add_noise(ft_tst_series, noise_level)


print('loading data...')
df = openAndHandleUnknowns('tcd-ml-1920-group-income-train.csv')
sub_df = openAndHandleUnknowns('tcd-ml-1920-group-income-test.csv')

print('initial preprocessing...')
# dropping duplicate columns
df.drop_duplicates(inplace=True)
# removing income outliers and negative valued income.
df = removeIncomeRows(df)

df.isnull().sum(axis=0)

print('filling NaN values...')
df = dfFillNaN(df)
sub_df = dfFillNaN(sub_df)

# applying log transform on 'Total Yearly Income [EUR]'
df['Total Yearly Income [EUR]'] = np.log(df['Total Yearly Income [EUR]'])

y = df['Total Yearly Income [EUR]']
instance = pd.DataFrame(sub_df['Instance'], columns=['Instance'])

# features being considered for linear regression
features = ['Year of Record', 'Housing Situation', 'Crime Level in the City of Employement',
            'Work Experience in Current Job [years]', 'Satisfation with employer',
            'Gender', 'Age', 'University Degree', 'Country', 'Size of City', 'Profession',
            # 'Wears Glasses', 'Hair Color', 'Body Height [cm]',
            'Yearly Income in addition to Salary (e.g. Rental Income)']

df = df[features]
sub_df = sub_df[features]

# Feature modifications
# Standard Scaling
print('Normalizing data...')
yor_scalar = pp.StandardScaler()
df['Year of Record'] = yor_scalar.fit_transform(df['Year of Record'].values.reshape(-1, 1))
sub_df['Year of Record'] = yor_scalar.transform(sub_df['Year of Record'].values.reshape(-1, 1))

age_scalar = pp.StandardScaler()
df['Age'] = age_scalar.fit_transform(df['Age'].values.reshape(-1, 1))
sub_df['Age'] = age_scalar.transform(sub_df['Age'].values.reshape(-1, 1))

# Target Encoding
df['Gender'], sub_df['Gender'] = target_encode(df['Gender'], sub_df['Gender'], y)

df['University Degree'], sub_df['University Degree'] = target_encode(df['University Degree'],
                                                                     sub_df['University Degree'], y)
# df['Hair Color'], sub_df['Hair Color'] = target_encode(df['Hair Color'], sub_df['Hair Color'], y)

df['Housing Situation'], sub_df['Housing Situation'] = target_encode(df['Housing Situation'],
                                                                     sub_df['Housing Situation'], y)
df['Satisfation with employer'], sub_df['Satisfation with employer'] = \
    target_encode(df['Satisfation with employer'], sub_df['Satisfation with employer'], y)

# replacing the a small number of least count group values to a common feature 'other'
countryList = df['Country'].unique()
countryReplaced = df.groupby('Country').count()
countryReplaced = countryReplaced[countryReplaced['Age'] < 3].index
df['Country'].replace(countryReplaced, 'other', inplace=True)

# Handling the 'other' encoding in Country Feature
testCountryList = sub_df['Country'].unique()
encodedCountries = list(set(countryList) - set(countryReplaced))
testCountryReplace = list(set(testCountryList) - set(encodedCountries))
sub_df['Country'] = sub_df['Country'].replace(testCountryReplace, 'other')

df['Country'], sub_df['Country'] = target_encode(df['Country'], sub_df['Country'], y)

# replacing the a small number of least count group values to a common feature 'other profession'
professionList = df['Profession'].unique()
professionReplaced = df.groupby('Profession').count()
professionReplaced = professionReplaced[professionReplaced['Age'] < 3].index
df['Profession'].replace(professionReplaced, 'other profession', inplace=True)

# Handling the 'other profession' encoding in Profession Feature
testProfessionList = sub_df['Profession'].unique()
encodedProfession = list(set(professionList) - set(professionReplaced))
testProfessionReplace = list(set(testProfessionList) - set(encodedProfession))
sub_df['Profession'] = sub_df['Profession'].replace(testProfessionReplace, 'other profession')

df['Profession'], sub_df['Profession'] = target_encode(df['Profession'], sub_df['Profession'], y)

print('Training Model!!')
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

params = {}
params['learning_rate'] = 0.001
params['boosting_type'] = 'gbdt'
params['metric'] = 'mae'
params['max_depth'] = 30
params['verbosity'] = -1

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)
model = lgb.train(params=params, train_set=train_data, num_boost_round=100000, valid_sets=[train_data, test_data],
                  verbose_eval=1000, early_stopping_rounds=500)

print('predicting Y...')
y_pred = model.predict(X_test)

# applying inverse log transform to get the actual values
y_pred = np.exp(y_pred)
y_test = np.exp(y_test)

print("MAE: %.2f" % mean_absolute_error(y_test, y_pred))
print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

##################################################################################################################
print('\n\nPredicting the output!')
y_sub = model.predict(sub_df)

# applying inverse log transform to get the actual values
y_sub = np.exp(y_sub)

income = pd.DataFrame(y_sub, columns=['Total Yearly Income [EUR]'])
ans = instance.join(income)

print('creating final csv...')
ans.to_csv('kaggle-output.csv', index=False)
