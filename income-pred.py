import os
import pandas as pd
import numpy as np
from scipy import stats
from sklearn import preprocessing as pp
from sklearn.ensemble import RandomForestRegressor as rfr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# reading data and handling unknowns
def openAndHandleUnknowns(fileName):
    return pd.read_csv(fileName, na_values={
        'Year of Record': [0, '#N/A', 'unknown'],
        'Gender': [0, '#N/A', 'unknown'],
        'Age': [0, '#N/A', 'unknown'],
        'Country': ['#N/A', 'unknown'],
        'Size of City': ['#N/A', 'unknown'],
        'Profession': ['#N/A'],
        'University Degree': [0, '#N/A', 'unknown'],
        'Wears Glasses': ['#N/A', 'unknown'],
        'Hair Color': [0, '#N/A', 'Unknown'],
        'Body Height [cm]': ['#N/A', 'unknown'],
        'Income in EUR': []
    })


# handling NaN
# TODO try different methods of interpolate to reduce RMSE
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.interpolate.html#pandas.DataFrame.interpolate
def dfFillNaN(df):
    df['Year of Record'] = np.floor(df['Year of Record'].interpolate(method='slinear'))
    df['Gender'].fillna('other', inplace=True)
    df['Age'] = np.floor(df['Age'].interpolate(method='slinear'))
    # different methods of fillna?
    df['Profession'].fillna(method='ffill', inplace=True)
    df['University Degree'].fillna(method='ffill', inplace=True)
    df['Hair Color'].fillna(method='ffill', inplace=True)
    return df


# Removing outliers using z-score
def dropNumericalOutliers(df, z_thresh=3):
    # Constrains will contain `True` or `False` depending on if it is a value below the threshold.
    constrains = df.select_dtypes(include=[np.number]) \
        .apply(lambda x: np.abs(stats.zscore(x)) < z_thresh) \
        .all(axis=1)
    # Drop (inplace) values set to be rejected
    df.drop(df.index[~constrains], inplace=True)
    return df


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


df = openAndHandleUnknowns('tcd ml 2019-20 income prediction training (with labels).csv')
sub_df = openAndHandleUnknowns('tcd ml 2019-20 income prediction test (without labels).csv')

df = dfFillNaN(df)
sub_df = dfFillNaN(sub_df)

df['Income in EUR'] = df['Income in EUR'].abs()
y = df['Income in EUR']
instance = pd.DataFrame(sub_df['Instance'], columns=['Instance'])

# features being considered for linear regression
features = ['Year of Record', 'Gender', 'Age', 'University Degree', 'Wears Glasses', 'Hair Color',
            'Body Height [cm]', 'Country', 'Size of City', 'Profession']

df = df[features + ['Income in EUR']]
sub_df = sub_df[features]

# Feature modifications
# Standard Scaling
yor_scalar = pp.StandardScaler()
df['Year of Record'] = yor_scalar.fit_transform(df['Year of Record'].values.reshape(-1, 1))

sub_df['Year of Record'] = yor_scalar.transform(sub_df['Year of Record'].values.reshape(-1, 1))

age_scalar = pp.StandardScaler()
df['Age'] = age_scalar.fit_transform(df['Age'].values.reshape(-1, 1))

sub_df['Age'] = age_scalar.transform(sub_df['Age'].values.reshape(-1, 1))

# Target Encoding
df['Gender'], sub_df['Gender'] = target_encode(df['Gender'], sub_df['Gender'], y)

df['University Degree'], sub_df['University Degree'] = target_encode(df['University Degree'], sub_df['University Degree'], y)

df['Hair Color'], sub_df['Hair Color'] = target_encode(df['Hair Color'], sub_df['Hair Color'], y)

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

del df['Income in EUR']
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)

model = rfr(n_estimators=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("RMSE: %.2f" % np.sqrt(mean_squared_error(y_test, y_pred)))
print('Variance score: %.2f' % r2_score(y_test, y_pred))

##################################################################################################################
y_sub = model.predict(sub_df)
income = pd.DataFrame(y_sub, columns=['Income'])
ans = instance.join(income)

ans.to_csv('kaggle-output.csv', index=False)
