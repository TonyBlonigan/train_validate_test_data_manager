import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DataManager():
    def __init__(self, df, y_col, val_prop, test_prop, x_scaler, y_scaler, split_on_date_col=None, no_scale_cols=None,
                 timesteps_col=None, sum_timeseries_y=False):
        """
        Take a cleansed data set in the form of a data frame and Split to train, test & validate sets, impute means for
        missing values, drop rows where no values is available for the y variable.

        :param df: data frame with generic 1-n index, categorical cols alread one hot encoded
        :param y_col: string name of column with y variable
        :param val_prop: float
        :param test_prop: float
        :param scaler:
        :param split_on_date_col: string, column name of date to split data on. If null, will be randomly split without
        :param no_scale_cols: list of columns you don't want scaled that are not one hot encoded or split_on_date_col
        :param timesteps_col: string, name of column with order of time steps as datetime.datetime()
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

        self.timesteps_col = timesteps_col
        if self.timesteps_col is not None:
            self.x_cats += [self.timesteps_col]
            self.x_non_cats.remove(self.timesteps_col)

        self.no_scale_cols = no_scale_cols
        if no_scale_cols is not None:
            self.x_cats += self.no_scale_cols
            for c in self.no_scale_cols:
                self.x_non_cats.remove(c)

        print('categorical and date columns:\n\t' + str(self.x_cats))
        if self.timesteps_col is not None: print('timesteps col:' + self.timesteps_col)

        # drop obs with na in y_col
        self.df = df.dropna(axis=0, how='any', subset=[y_col])

        # sort by timesteps if using timesteps data, and drop any unlabled timesteps, make sure each split_on_date
        # has same amount of steps
        if self.timesteps_col is not None:
            self.T_timesteps = None
            self.timeseries_preprocessing()

        self.train_prop = 1 - val_prop - test_prop
        self.val_prop = val_prop
        self.test_prop = test_prop

        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.x_train = None
        self.y_train = None
        self.x_val = None
        self.y_val = None
        self.x_test = None
        self.y_test = None
        self.set_train_val_test_sets()

        # add ts steps since last obs
        if timesteps_col is not None:
            for data_set in (self.x_train, self.x_val, self.x_test):
                self.add_steps_since_last_timestep(data_set)
            self.x_cols.append('t_since_last_ts')
            self.x_non_cats.append('t_since_last_ts')

        # keep track of dates
        if self.split_on_date_col is not None:
            self.x_train_dts = self.drop_split_on_date_col(self.x_train, self.split_on_date_col)
            self.x_val_dts = self.drop_split_on_date_col(self.x_val, self.split_on_date_col)
            self.x_test_dts = self.drop_split_on_date_col(self.x_test, self.split_on_date_col)
            self.x_cols.remove(self.split_on_date_col)
            self.x_cats.remove(self.split_on_date_col)

        if self.timesteps_col is not None:
            self.x_train_ts = self.drop_split_on_date_col(self.x_train, self.timesteps_col)
            self.x_val_ts = self.drop_split_on_date_col(self.x_val, self.timesteps_col)
            self.x_test_ts = self.drop_split_on_date_col(self.x_test, self.timesteps_col)
            self.x_cols.remove(self.timesteps_col)
            self.x_cats.remove(self.timesteps_col)

        self.impute_means()
        self.scale_train_val_test_sets()

        if self.timesteps_col is not None:
            self.sum_timeseries_y = sum_timeseries_y
            self.to_timeseries()

    def drop_split_on_date_col(self, df, col):
        dts = df[col]
        df.drop(columns=col, inplace=True)
        return dts

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
            self.x_train, self.x_val, self.y_train, self.y_val = self.train_test_split_date(x=self.df[self.x_cols],
                                                                                            y=self.df[self.y_col],
                                                                                            test_size=val_plus_test_prop)

            self.x_val, self.x_test, self.y_val, self.y_test = self.train_test_split_date(x=self.x_val,
                                                                                          y=self.y_val,
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

        max_test_dt = (np.abs(val_freq - (1-test_size))).idxmin()

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

        if nans > 0:
            for c in self.x_non_cats:
                # check whether this column has nans in train, val or test
                nans = 0
                for data_set in (self.x_train, self.x_val, self.x_test):
                    nans += data_set[c].isna().sum().sum()

                if nans > 0:
                    print('adding nan ind and imputing means for ' + c)
                    m = np.mean(self.x_train[c])
                    if m == np.nan: m = 0
                    for data_set in (self.x_train, self.x_val, self.x_test):
                        isna_mask = pd.isna(data_set[c])
                        data_set.loc[isna_mask, c] = m
                        # add nan indicator column
                        nan__c = 'nan_' + c
                        data_set.loc[~isna_mask, nan__c] = 0
                        data_set.loc[isna_mask, nan__c] = 1
                    self.x_cols.append(nan__c)
                    self.x_cats.append(nan__c)

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

        self.x_cols = self.x_non_cats + self.x_cats

        return x, y

    def add_steps_since_last_timestep(self, data_set):
        data_set['t_since_last_ts'] = (data_set[self.split_on_date_col] - data_set[self.timesteps_col]) \
            .apply(lambda x: x.days)

        data_set['t_since_last_ts'] = data_set['t_since_last_ts'].diff() * -1

        split_on_date_col_changed_values = data_set[self.split_on_date_col] != data_set[self.split_on_date_col].shift()

        data_set.loc[split_on_date_col_changed_values, 't_since_last_ts'] = 0

    def timeseries_preprocessing(self):
        # drop unlabeled timesteps
        self.df = self.df \
            .dropna(axis=0, how='any', subset=[self.timesteps_col]) \
            .sort_values([self.split_on_date_col, self.timesteps_col])

        # drop extra timesteps if some obs have more timesteps
        T_by_i = self.df[self.split_on_date_col].value_counts()
        self.T_timesteps = T_by_i.min()
        n_i_dropping_steps = sum(T_by_i > self.T_timesteps)
        print('max timestep count = ' + str(T_by_i.max()) +
              '\nmin timestep count = ' + str(self.T_timesteps) +
              '\ndropping timesteps from ' + str(n_i_dropping_steps) + ' of ' + str(len(T_by_i)) +
              ' with steps begining before T - ' + str(self.T_timesteps))

        split_ts_cols = [self.split_on_date_col, self.timesteps_col]
        self.df = self.df \
            .sort_values(split_ts_cols) \
            .groupby(self.split_on_date_col) \
            .tail(self.T_timesteps)

    def to_timeseries(self):
        """
        convert 2d data set to 3d timeseries data (observations, timesteps, variables)
        :param data_set: np.array like self.x_train
        :param dts: np.array of datetime like self.x_train_dts of observation dates
        :param ts: np.array of datetime like self.x_train_ts of timesteps
        :return: updated data_set, dts & ts
        """
        self.x_train = self.x_train.reshape((-1, self.T_timesteps, self.x_train.shape[1]))
        self.y_train = self.y_train.reshape((-1, self.T_timesteps, self.y_train.shape[1]))
        self.x_train_dts = self.x_train_dts.unique()
        self.x_train_ts = self.x_train_ts.values.reshape((-1, self.T_timesteps))

        self.x_val = self.x_val.reshape((-1, self.T_timesteps, self.x_val.shape[1]))
        self.y_val = self.y_val.reshape((-1, self.T_timesteps, self.y_val.shape[1]))
        self.x_val_dts = self.x_val_dts.unique()
        self.x_val_ts = self.x_val_ts.values.reshape((-1, self.T_timesteps))

        self.x_test = self.x_test.reshape((-1, self.T_timesteps, self.x_test.shape[1]))
        self.y_test = self.y_test.reshape((-1, self.T_timesteps, self.y_test.shape[1]))
        self.x_test_dts = self.x_test_dts.unique()
        self.x_test_ts = self.x_test_ts.values.reshape((-1, self.T_timesteps))

        if self.sum_timeseries_y:
            self.y_train = np.sum(self.y_train, axis=1).reshape((self.y_train.shape[0], 1))
            self.y_val = np.sum(self.y_val, axis=1).reshape((self.y_val.shape[0], 1))
            self.y_test = np.sum(self.y_test, axis=1).reshape((self.y_test.shape[0], 1))

if __name__ == '__main__':
    from sklearn.preprocessing import MinMaxScaler

    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    df = pd.DataFrame({'a': [1, 2, 2, 3, 2, 1],
                       'cat': [0, 0, 1, 1, 0, 1],
                       'c': [1.2, np.nan, 2.2, np.nan, 4.2, 5.2],
                       'd': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                       'y': [1, 3.5, np.nan, 1.1, 3.2, 2.3],
                       'the_dates': pd.date_range(datetime.date(2018, 1, 1), datetime.datetime(2018, 1, 6)),
                       'observed_on_date': pd.date_range(datetime.date(2017, 12, 30), datetime.datetime(2018, 1, 4))})
    
    dm = DataManager(df, y_col='y', val_prop=.2, test_prop=.2,
                     x_scaler=x_scaler, y_scaler=y_scaler,
                     split_on_date_col='the_dates', no_scale_cols=['observed_on_date'])
    
    print(dm.x_train)
    print(dm.y_train)
    print()
    print(dm.x_val)
    print(dm.y_val)
    print()
    print(dm.x_test)
    print(dm.y_test)
