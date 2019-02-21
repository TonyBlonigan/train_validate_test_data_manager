import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager():
    def __init__(self, df, y_col, val_prop, test_prop, x_scaler, y_scaler, split_on_date_col=None):
        """

        :param df: data frame with generic 1-n index, categorical cols alread one hot encoded
        :param y_col: string name of column with y variable
        :param val_prop: float
        :param test_prop: float
        :param scaler:
        :param split_on_date_col: string, column name of date to split data on. If null, will be randomly split without
        considering date of observation
        """
        self.y_col = y_col

        # figure out what columns are categorical
        self.x_cats = [c for c in df.columns if (len(df[c].unique()) == 2) and (0 in df[c].unique()) and (1 in df[c])]
        self.x_cols = [c for c in df if c != y_col]
        self.x_non_cats = [c for c in self.x_cols if c not in self.x_cats]
        self.x_cols = self.x_non_cats + self.x_cats

        self.split_on_date_col = split_on_date_col
        if self.split_on_date_col is not None:
            self.x_cats += [self.split_on_date_col]
            self.x_non_cats.remove(self.split_on_date_col)
        print('categorical and date columns:\n\t' + str(self.x_cats))

        # drop obs with na in y_col
        self.df = df.dropna(axis=0, how='any', subset=[y_col])

        self.train_prop = 1 - val_prop - test_prop
        self.val_prop = val_prop
        self.test_prop = test_prop

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.x_cols
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.set_train_val_test_sets()
        self.impute_means()
        self.scale_train_val_test_sets()

    def set_train_val_test_sets(self):
        val_plus_test_prop = self.val_prop + self.test_prop
        relative_test_prop = self.test_prop / val_plus_test_prop

        if self.split_on_date_col is None:
            self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(self.df[self.x_cols], self.df[self.y_col],
                                                                                  test_size=val_plus_test_prop,
                                                                                  shuffle=True,
                                                                                  seed=123)

            self.x_val, self.x_test, self.y_val, self.y_test = train_test_split(self.x_val, self.y_val,
                                                                                test_size=relative_test_prop,
                                                                                shuffle=True,
                                                                                seed=123)
        else:
            self.x_train, self.x_val, self.y_train, self.y_val = self.train_test_split_date(self.df[self.x_cols],
                                                                                            self.df[self.y_col],
                                                                                            test_size=val_plus_test_prop)

            self.x_val, self.x_test, self.y_val, self.y_test = self.train_test_split_date(self.x_val,
                                                                                          self.y_val,
                                                                                          test_size=relative_test_prop)

    def overwrite_null_training_col(self):
        for c in self.x_cols:
            if pd.isna(self.x_train[c]).all():
                self.x_train[c] = 0
                self.x_val[c] = 0
                self.x_test[c] = 0

    def train_test_split_date(self, x, y, test_size):
        val_freq = x[self.split_on_date_col].value_counts(normalize=True).sort_index()

        val_freq = val_freq.cumsum()

        max_test_dt = (np.abs(val_freq - test_size)).idxmin()

        test_date_mask = x[self.split_on_date_col] <= max_test_dt

        x_a = x[test_date_mask]
        x_b = x[~test_date_mask]
        y_a = y[test_date_mask]
        y_b = y[~test_date_mask]

        return x_a, x_b, y_a, y_b

    def impute_means(self):
        # if no nans in data set, skip this entire thing
        nans = 0
        for data_set in (self.x_train, self.x_val, self.x_test):
            nans += data_set.isna().sum().sum()

        if nans:
            for c in self.x_non_cats:
                m = np.mean(self.x_train[c])
                for data_set in (self.x_train, self.x_val, self.x_test):
                    data_set.loc[pd.isna(data_set[c]), c] = m

    def scale_train_val_test_sets(self):
        """
        scale the train and validation sets
        :return: None
        """
        self.overwrite_null_training_col()

        self.x_train, self.y_train = self.scale_x_and_y(self.x_train, self.y_train, fit_scaler=True)

        self.x_val, self.y_val = self.scale_x_and_y(self.x_val, self.y_val)

        self.x_test, self.y_test = self.scale_x_and_y(self.x_test, self.y_test)

    def scale_x_and_y(self, x, y, fit_scaler=False):
        if fit_scaler:
            x_non_cats = self.x_scaler.fit_transform(x[self.x_non_cats])
            y = self.y_scaler.fit_transform(y.to_frame())
        else:
            x_non_cats = self.x_scaler.transform(x[self.x_non_cats])
            y = self.y_scaler.transform(y.to_frame())

        x_cats = x[self.x_cats]

        x = np.concatenate((x_non_cats, x_cats), axis=1)

        return x, y


if __name__ == '__main__':
    from sklearn.preprocessing import MinMaxScaler

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