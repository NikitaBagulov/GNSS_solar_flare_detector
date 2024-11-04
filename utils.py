import datetime
import math
from dateutil import tz

from numpy.typing import NDArray
import h5py


RE_meters = 6371000

TIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
_UTC = tz.gettz('UTC')
DAYS = 365.25
TILT = 23.44
MAG1 = 9
LON_DEGREES_HOUR = 15.

def get_latlon(time):
        delta = time - datetime.datetime(time.year, 1, 1, 0, 0, 0)
        doy = delta.days
        ut_hour = time.hour + time.minute / 60. + time.second / (60. * 24.)
        lat = - TILT * math.cos(2 * math.pi * ((doy + MAG1)) / DAYS)
        lon = (12.0 - ut_hour) * LON_DEGREES_HOUR
        return lat, lon

def retrieve_data(file) -> dict[datetime.datetime, NDArray]:
        f_in = h5py.File(file, 'r')
        data = {}
        times = list(f_in['data'])[:]
        
        
        for str_time in times:
            time = datetime.datetime.strptime(str_time, TIME_FORMAT).replace(tzinfo=tz.gettz('UTC'))
            data[time] = f_in['data'][str_time][:]

        return data