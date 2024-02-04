import pandas as pd
import pytz
import datetime as dt


class DefaultValueFiller:
    """
    Default_value_mode:
    0: Calculate default value based on the entire dataset.
    1: Calculate default value based on monthly data.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            feature_names=None,
            default_value_mode: int = 0,
            date_column: str = 'timestamp',
            freq: str = '5min'):

        feature_names = [] if feature_names is None else feature_names

        self.df = df
        self.feature_names = feature_names
        self.default_value_mode = default_value_mode
        self.datetime_column = date_column
        self.freq = freq
        self.feature_data = self.get_feature_data()
        # self.new_dataset = self.fill_missing_value()

    def format_date(self):
        self.df[self.datetime_column] = pd.to_datetime(self.df[self.datetime_column])
        df_local = self.df[self.datetime_column].dt.tz_localize(None).dt.floor('min')

        return df_local

    def transfer_time_zone(self):
        self.df[self.datetime_column] = pd.to_datetime(self.df[self.datetime_column])
        df_utc = self.df[self.datetime_column].dt.tz_localize('UTC')
        df_local = df_utc.dt.tz_convert(pytz.timezone('US/Eastern')).dt.tz_localize(None).dt.floor('min')

        return df_local

    def get_feature_data(self):
        # Time zone transfer from UTC to local ('US/Eastern)
        # *******************************************************************************
        # date_local = self.transfer_time_zone()
        date_local = self.format_date()

        # Get data from specific column
        # *******************************************************************************
        feature_data = pd.concat([date_local, self.df[self.feature_names]], axis=1)
        feature_data['weekday'] = feature_data[self.datetime_column].dt.weekday

        return feature_data

    @staticmethod
    def get_date_range(data: pd.DataFrame, date_column: str = 'data_time'):
        dt_min = data[date_column].min()
        dt_max = data[date_column].max()
        year_range = range(dt_min.year, dt_max.year + 1)

        months_range = {}
        days_range = {}

        for year in year_range:
            month_min = data[data[date_column].dt.year == year][date_column].dt.month.min()
            month_max = data[data[date_column].dt.year == year][date_column].dt.month.max()
            day_min = data[(data[date_column].dt.year == year) &
                           (data[date_column].dt.month == month_min)][date_column].dt.day.min()
            day_max = data[(data[date_column].dt.year == year) &
                           (data[date_column].dt.month == month_max)][date_column].dt.day.max()
            months_range[year] = range(month_min, month_max + 1)
            days_range[year] = (day_min, day_max)

        return year_range, months_range, days_range

    def fill_missing_value(self):

        data = self.get_feature_data()
        # Construct new dataframe without missing value for each timestep
        # *******************************************************************************
        hr_interval = None
        min_interval = None
        if 'min' in self.freq:
            min_interval = int(self.freq.split('min')[0])
        elif 'h' in self.freq:
            hr_interval = int(self.freq.split('h')[0])
        else:
            pass

        datetimes = []
        year_range, months_range, days_range = DefaultValueFiller.get_date_range(data, date_column=self.datetime_column)

        # start_day, end_day = dt_min.day, dt_max.day
        num_days_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        for year in year_range:
            month_range = months_range[year]
            if len(months_range[year]) == 1:
                for day in range(1, days_range[year][1] + 1):
                    if min_interval is not None:
                        for hour in range(24):
                            for minute in range(0, 60, min_interval):
                                datetimes.append(
                                    dt.datetime(year=year, month=month_range[0], day=day, hour=hour, minute=minute))
                    else:
                        if hr_interval is not None:
                            for hour in range(0, 24, hr_interval):
                                datetimes.append(
                                    dt.datetime(year=year, month=month_range[0], day=day, hour=hour, minute=0))
            else:
                for i, month in enumerate(month_range):
                    if i == 0:
                        for day in range(days_range[year][0], num_days_month[month - 1] + 1):
                            if min_interval is not None:
                                for hour in range(24):
                                    for minute in range(0, 60, min_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour,
                                                        minute=minute))
                            else:
                                if hr_interval is not None:
                                    for hour in range(0, 24, hr_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour, minute=0))
                    elif i == len(month_range) - 1:
                        for day in range(1, days_range[year][1] + 1):
                            if min_interval is not None:
                                for hour in range(24):
                                    for minute in range(0, 60, min_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour,
                                                        minute=minute))
                            else:
                                if hr_interval is not None:
                                    for hour in range(0, 24, hr_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour, minute=0))
                    else:
                        for day in range(1, num_days_month[month - 1] + 1):
                            if min_interval is not None:
                                for hour in range(24):
                                    for minute in range(0, 60, min_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour,
                                                        minute=minute))
                            else:
                                if hr_interval is not None:
                                    for hour in range(0, 24, hr_interval):
                                        datetimes.append(
                                            dt.datetime(year=year, month=month, day=day, hour=hour, minute=0))

        new_df = pd.DataFrame(pd.to_datetime(datetimes), columns=[self.datetime_column])
        new_df['weekday'] = new_df[self.datetime_column].dt.weekday

        for feature in self.feature_names:
            if self.default_value_mode == 0:
                default = self.calc_default_value(feature)
                filled_data = []
                for date in datetimes:
                    value = self.feature_data[(self.feature_data[self.datetime_column] == date)][feature].values

                    if len(value) == 0:
                        weekday = date.weekday()
                        if weekday in [0, 1, 2, 3, 4]:
                            value = default[0][date.hour][date.minute]
                        elif weekday in [5, 6]:
                            value = default[1][date.hour][date.minute]

                        filled_data.append(value)
                    else:
                        filled_data.append(value[0])
            else:
                default = self.calc_default_value_monthly(feature)
                filled_data = []
                for date in datetimes:
                    value = self.feature_data[(self.feature_data[self.datetime_column] == date)][feature].values

                    if len(value) == 0:
                        current_year = date.year
                        current_month = date.month
                        weekday = date.weekday()
                        if weekday in [0, 1, 2, 3, 4]:
                            value = default[current_year][current_month][0][date.hour][date.minute]
                        elif weekday in [5, 6]:
                            value = default[current_year][current_month][1][date.hour][date.minute]
                        filled_data.append(value)
                    else:
                        filled_data.append(value[0])

            # Fill the strange zero value:
            DefaultValueFiller.fill_zero_value(filled_data)
            DefaultValueFiller.fill_strange_value(filled_data)

            new_df[feature] = filled_data
            # new_df.drop(['weekday'], axis=1, inplace=True)

        return new_df

    @staticmethod
    def fill_zero_value(data):
        """
        Replace zero value in the input list with its previous value.
        """
        for i in range(len(data)):
            if data[i] == 0:
                if i == 0:
                    data[i] = data[i + 1]
                else:
                    data[i] = data[i - 1]

    @staticmethod
    def fill_strange_value(data):
        """
        Replace strange high or low value in the input list with modified ratio of its previous value.
        """
        delta_threshold = 0.8
        modified_ratio = 0.5
        for i in range(len(data)):
            if data[i] != 0 and data[i - 1] != 0:
                if data[i] > data[i - 1] and 1 - (data[i] / data[i - 1]) > delta_threshold:
                    data[i] = data[i - 1] * modified_ratio
                elif data[i] < data[i - 1] and 1 - (data[i - 1] / data[i]) > delta_threshold:
                    data[i] = data[i - 1] * modified_ratio
                else:
                    pass
            else:
                pass

    def calc_default_value(self, column_name):
        """
            Calculate average value of every minute in a day by weekday (from Monday to Sunday).
            Use the calculated value to fill empty/missing value in the dataset.
        """
        hours = range(24)
        minutes = range(60)
        default_values = {}

        weekdays = {}
        weekends = {}

        for hour in hours:
            hours_wday = []
            hours_wend = []
            for minute in minutes:
                value_wday = self.feature_data[
                    ((self.feature_data['weekday'] == 0) |
                     (self.feature_data['weekday'] == 1) |
                     (self.feature_data['weekday'] == 2) |
                     (self.feature_data['weekday'] == 3) |
                     (self.feature_data['weekday'] == 4)) &
                    (self.feature_data[self.datetime_column].dt.hour == hour) &
                    (self.feature_data[self.datetime_column].dt.minute == minute)][column_name].mean()

                value_wend = self.feature_data[
                    ((self.feature_data['weekday'] == 5) |
                     (self.feature_data['weekday'] == 6)) &
                    (self.feature_data[self.datetime_column].dt.hour == hour) &
                    (self.feature_data[self.datetime_column].dt.minute == minute)][
                    column_name].mean()

                hours_wday.append(value_wday)
                hours_wend.append(value_wend)

            weekdays[hour] = hours_wday
            weekends[hour] = hours_wend

        default_values[0] = weekdays
        default_values[1] = weekends

        return default_values

    def calc_default_value_monthly(self, column_name):
        """
            Calculate average value of every minute in a day by monthly average.
            Use the calculated value to fill empty/missing value in the dataset.
        """
        days_threshold = 3

        hours = range(24)
        minutes = range(60)
        default_values = {}

        data = self.get_feature_data()
        # Construct new dataframe without missing value for each timestep
        # *******************************************************************************
        year_range, months_range, days_range = DefaultValueFiller.get_date_range(data, date_column=self.datetime_column)

        for year in year_range:
            year_values = {}
            month_range = months_range[year]
            day_range = days_range[year]

            for month in month_range:
                # Calculate the number of days of the last month of the current year:
                days_of_last_month = day_range[1]

                weekdays = {}
                weekends = {}
                month_values = {}
                for hour in hours:
                    hours_wday = []
                    hours_wend = []
                    for minute in minutes:

                        # If number of days in this month is not enough (less than threshold) to
                        # calculate monthly average, then use the previous month to do calculation.
                        current_year = year
                        current_month = month
                        if days_of_last_month < days_threshold:
                            if month == 1:
                                current_year = year - 1
                                current_month = 12
                            else:
                                current_month = month - 1
                        else:
                            pass

                        value_wday = data[(data[self.datetime_column].dt.year == current_year) &
                                          (data[self.datetime_column].dt.month == current_month) &
                                          ((data[self.datetime_column].dt.weekday == 0) |
                                           (data[self.datetime_column].dt.weekday == 1) |
                                           (data[self.datetime_column].dt.weekday == 2) |
                                           (data[self.datetime_column].dt.weekday == 3) |
                                           (data[self.datetime_column].dt.weekday == 4)) &
                                          (data[self.datetime_column].dt.hour == hour) &
                                          (data[self.datetime_column].dt.minute == minute)][column_name].mean()

                        value_wend = data[(data[self.datetime_column].dt.year == current_year) &
                                          (data[self.datetime_column].dt.month == current_month) &
                                          ((data[self.datetime_column].dt.weekday == 5) |
                                           (data[self.datetime_column].dt.weekday == 6)) &
                                          (data[self.datetime_column].dt.hour == hour) &
                                          (data[self.datetime_column].dt.minute == minute)][column_name].mean()

                        hours_wday.append(value_wday)
                        hours_wend.append(value_wend)

                    weekdays[hour] = hours_wday
                    weekends[hour] = hours_wend

                month_values[0] = weekdays
                month_values[1] = weekends

                year_values[month] = month_values
            default_values[year] = year_values

        return default_values
