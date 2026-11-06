"""Simplified calendar for the energy package.

All calendar conventions live in this module, so that a switch to a
realistic calendar only requires changes here.

Current convention:
    * a year has 12 months of 30 days each (360 days per year)
    * every day has 24 hours, so every month has 720 hours
    * month index m counts months from the start of the simulation and
      covers the model time interval [m/12, (m+1)/12) in years
"""

import numpy as np

__all__ = [
    "DAYS_PER_YEAR",
    "MONTHS_PER_YEAR",
    "month_start",
    "month_end",
    "days_in_month",
    "hours_in_month",
    "day_to_time",
    "has_started",
]

MONTHS_PER_YEAR = 12
DAYS_PER_YEAR = 360
HOURS_PER_DAY = 24

# Tolerance for comparing model times, to absorb floating point errors
# such as 12 * (90 / 360) evaluating to 2.9999999999999996
_TIME_TOLERANCE = 1e-9


def month_start(month):
    """Model time in years at which the given month starts.

    Works for a single month index or an array of month indices.
    """
    return month / MONTHS_PER_YEAR


def month_end(month):
    """Model time in years at which the given month ends."""
    return month_start(month + 1)


def days_in_month(month):
    """Number of delivery days in the given month.

    Returns an int for a scalar month and an integer array of the same
    shape for an array of months.
    """
    days = DAYS_PER_YEAR // MONTHS_PER_YEAR
    if np.ndim(month) == 0:
        return days
    return np.full(np.shape(month), days, dtype=int)


def hours_in_month(month):
    """Number of delivery hours in the given month (scalar or array)."""
    return HOURS_PER_DAY * days_in_month(month)


def day_to_time(day):
    """Convert a day count since the simulation start into model time in years."""
    return day / DAYS_PER_YEAR


def has_started(month, t) -> bool:
    """Whether the given month has started at model time t.

    A month starting exactly at t counts as started, so it is settled on
    its first day and no longer part of the open position.
    """
    return month_start(month) <= t + _TIME_TOLERANCE
