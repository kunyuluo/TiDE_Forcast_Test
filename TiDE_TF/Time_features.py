import numpy as np
import pandas as pd
from pandas.tseries.holiday import EasterMonday, GoodFriday, USColumbusDay, USLaborDay, USMartinLutherKingJr
from pandas.tseries.holiday import USMemorialDay, USPresidentsDay, USThanksgivingDay
from pandas.tseries.holiday import Holiday, SU, TH
from pandas.tseries.offsets import Easter, Day, DateOffset
from sklearn.preprocessing import StandardScaler

max_window = 183 + 17

EasterSunday = Holiday("Easter Sunday", month=1, day=1, offset=[Easter(), Day(0)])
NewYearsDay = Holiday("New Years Day", month=1, day=1)
SuperBowl = Holiday(
    "Superbowl", month=2, day=1, offset=DateOffset(weekday=SU(1))
)
MothersDay = Holiday(
    "Mothers Day", month=5, day=1, offset=DateOffset(weekday=SU(2))
)
IndependenceDay = Holiday("Independence Day", month=7, day=4)
ChristmasEve = Holiday("Christmas", month=12, day=24)
ChristmasDay = Holiday("Christmas", month=12, day=25)
NewYearsEve = Holiday("New Years Eve", month=12, day=31)
BlackFriday = Holiday(
    "Black Friday",
    month=11,
    day=1,
    offset=[pd.DateOffset(weekday=TH(4)), Day(1)],
)
CyberMonday = Holiday(
    "Cyber Monday",
    month=11,
    day=1,
    offset=[pd.DateOffset(weekday=TH(4)), Day(4)],
)

holidays = [
    EasterMonday,
    # GoodFriday,
    # USColumbusDay,
    # USLaborDay,
    # USMartinLutherKingJr,
    # USMemorialDay,
    # USPresidentsDay,
    # USThanksgivingDay,
    # EasterSunday,
    # NewYearsDay,
    # SuperBowl,
    # MothersDay,
    # IndependenceDay,
    # ChristmasEve,
    # ChristmasDay,
    # NewYearsEve,
    # BlackFriday,
    # CyberMonday,
]


def distance_to_holiday(holiday):
    def distance_to_day(index):
        holiday_date = holiday.dates(
            index - pd.Timedelta(days=max_window),
            index + pd.Timedelta(days=max_window)
        )
        assert (len(holiday_date) != 0), print('No closest holiday for the date index {index} found.')
        return (index - holiday_date[0]).days

    return distance_to_day


class TimeCovariates():
    """Extract all time covariates except for holidays."""

    def __init__(self, datetimes, normalized=True, holiday=False):
        self.dti = datetimes
        self.normalized = normalized
        self.holiday = holiday

    def _minute_of_hour(self):
        minutes = np.array(self.dti.minute, dtype=np.float32)
        if self.normalized:
            minutes = minutes / 59.0 - 0.5
        return minutes

    def _hour_of_day(self):
        hours = np.array(self.dti.hour, dtype=np.float32)
        if self.normalized:
            hours = hours / 23.0 - 0.5
        return hours

    def _day_of_week(self):
        day_week = np.array(self.dti.dayofweek, dtype=np.float32)
        if self.normalized:
            day_week = day_week / 6.0 - 0.5
        return day_week

    def _day_of_month(self):
        day_month = np.array(self.dti.day, dtype=np.float32)
        if self.normalized:
            day_month = day_month / 30.0 - 0.5
        return day_month

    def _day_of_year(self):
        day_year = np.array(self.dti.dayofyear, dtype=np.float32)
        if self.normalized:
            day_year = day_year / 364.0 - 0.5
        return day_year

    def _month_of_year(self):
        month_year = np.array(self.dti.month, dtype=np.float32)
        if self.normalized:
            month_year = month_year / 11.0 - 0.5
        return month_year

    def _week_of_year(self):
        week_year = np.array(self.dti.strftime("%U").astype(int), dtype=np.float32)
        if self.normalized:
            week_year = week_year / 51.0 - 0.5
        return week_year

    def _get_holidays(self):
        dti_series = self.dti.to_series()
        hol_variates = np.vstack(
            [
                dti_series.apply(distance_to_holiday(h)).values
                for h in holidays
            ]
        )
        # hol_variates is (num_holiday, num_time_steps), the normalization should be
        # performed in the num_time_steps dimension.
        return StandardScaler().fit_transform(hol_variates.T).T

    def get_covariates(self):
        """Get all time covariates."""
        moh = self._minute_of_hour().reshape(1, -1)
        hod = self._hour_of_day().reshape(1, -1)
        dom = self._day_of_month().reshape(1, -1)
        dow = self._day_of_week().reshape(1, -1)
        doy = self._day_of_year().reshape(1, -1)
        moy = self._month_of_year().reshape(1, -1)
        woy = self._week_of_year().reshape(1, -1)

        all_covs = [moh, hod, dom, dow, doy, moy, woy]
        columns = ["moh", "hod", "dom", "dow", "doy", "moy", "woy"]
        if self.holiday:
            hol_covs = self._get_holidays()
            all_covs.append(hol_covs)
            columns += [f"hol_{i}" for i in range(len(holidays))]

        return pd.DataFrame(
            data=np.vstack(all_covs).transpose(),
            columns=columns,
            index=self.dti,
        )
