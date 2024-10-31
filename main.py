from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge

import pandas as pd

import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack


def initialize(df: pd.DataFrame):
    df['FullDescription'] = df['FullDescription'].str.lower()
    df['FullDescription'] = df['FullDescription'].replace("[^A-Za-z0-9]", ' ', regex=True)
    return df


def fill(df: pd.DataFrame):
    df['LocationNormalized'].fillna('nan', inplace=True)
    df['ContractTime'].fillna('nan', inplace=True)
    return df

"""
Here starts the TRAIN block 
"""
train_data = pd.read_csv("data/salary-train.csv")
df_train = initialize(train_data)

tfidf_vec = TfidfVectorizer(min_df=5)
data = tfidf_vec.fit_transform(df_train['FullDescription'])

df_train = fill(df_train)
y_train = train_data['SalaryNormalized']

enc = DictVectorizer()
categ = enc.fit_transform(df_train[['LocationNormalized', 'ContractTime']].to_dict('records'))
x_train = hstack([data, categ])

ridge = Ridge(alpha=1, random_state=42)
ridge.fit(x_train, y_train)

"""
Here starts the TEST block 
"""
test_data = pd.read_csv("data/salary-test-mini.csv")
df_test = initialize(test_data)
dt = tfidf_vec.transform(df_test['FullDescription'])

df_test = fill(df_test)
categ2 = enc.transform(df_test[['LocationNormalized', 'ContractTime']].to_dict('records'))
x_test = hstack([dt, categ2])

y_test = ridge.predict(x_test)
print(y_test)
