# Common Train Validate Split Task Helper

This is designed to take a cleansed data set in the form of a data frame and
1. Split to train, test & validate sets
    - if you provide a `split_on_date_col` will split chronologicaly, otherwise will split randomly
2. Impute means for missing values
    - if no non-na values exist in training set for a given column, sets train, val and test set values for that column to 0
3. Drops rows where no values is available for the y variable

The processing doesn't happen in the order outlined above.


## Usage
```
from sklearn.preprocessing import MinMaxScaler
import dataframemanager

x_scaler = MinMaxScaler()
y_scaler = MinMaxScaler()

df = pd.DataFrame({'a': [1, 2, 2, 3, 2, 1],
                   'cat': [0, 0, 1, 1, 0, 1],
                   'c': [1.2, np.nan, 2.2, np.nan, 4.2, 5.2],
                   'd': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                   'y': [1, 3.5, np.nan, 1.1, 3.2, 2.3],
                   'the_dates': pd.date_range(datetime.date(2018, 1, 1), datetime.datetime(2018, 1, 6))})


dm = DataManager(df, y_col='y', val_prop=.2, test_prop=.2,
                 x_scaler=x_scaler, y_scaler=y_scaler,
                 split_on_date_col='the_dates')

print(dm.x_train)
print(dm.y_train)
print()
print(dm.x_val)
print(dm.y_val)
print()
print(dm.x_test)
print(dm.y_test)
```

## Results:
```
categorical and date columns:
	['cat', 'the_dates']
     a    c   d  the_dates  cat
0  1.0  1.2 NaN 2018-01-01    0
1  2.0  1.2 NaN 2018-01-02    0
0    1.0
1    3.5
Name: y, dtype: float64
     a    c   d  the_dates  cat
3  3.0  1.2 NaN 2018-01-04    1
4  2.0  4.2 NaN 2018-01-05    0
3    1.1
4    3.2
Name: y, dtype: float64
     a    c   d  the_dates  cat
5  1.0  5.2 NaN 2018-01-06    1
5    2.3
Name: y, dtype: float64
```

