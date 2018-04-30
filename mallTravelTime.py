import os.path as opath
import os
from functools import reduce
from datetime import datetime, date
from dateutil import tz
import pytz
import csv
import time
import pandas as pd
import googlemaps
#
from init_project import dpath

TARGET_MALLS = ['IMM', 'Tampines Mall', '313',
                'Bukit Panjang Plaza', 'Tiong Bahru Plaza', 'Payer Laber Quarter']
TARGET_HOURS = list(range(7, 12))


MIN60SEC = 60 * 60
MIN15SEC = 15 * 60
MIN15 = 15
N_TS_HOUR = int(MIN60SEC / MIN15SEC)
TT_CSV = reduce(opath.join, [opath.expanduser('~'), 'Dropbox', 'Data', '_mallTravelTime_googleMaps.csv'])
sce_dpath = '_scenario'
if not opath.exists(sce_dpath):
    os.mkdir(sce_dpath)


get_loc_dt = lambda loc: datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(tz.gettz(loc))


def dataCollection():
    ifpath = opath.join(dpath['geo'], 'mall-data.csv')
    malls = {}
    with open(ifpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            name = row['name']
            if name in TARGET_MALLS:
                lat, lon = map(float, [row[cn] for cn in ['latitude', 'longitude']])
                malls[name] = (lat, lon)
    if not opath.exists(TT_CSV):
        with open(TT_CSV, 'w') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_header = [
                'fromMall', 'toMall', 'duration',
                'year', 'month', 'day', 'dow', 'hour', 'minute']
            writer.writerow(new_header)
    #
    googleKey1 = 'AIzaSyAQYLeLHyJvNVC7uIbHmnvf7x9XC6murmk'
    googleKey2 = 'AIzaSyDCiqj9QQ-lXWGmzxXM0j-Gbeo_BRlsd0g'
    googleKey = googleKey2
    while True:
        try:
            get_duration_googleAPI(googleKey, malls)
        except:
            print('')
            print('Current key', googleKey)
            print('!!!!!!!!!!! change key !!!!!!!!!!!')
            googleKey = googleKey2 if googleKey == googleKey1 else googleKey1
            print('Changed key', googleKey)
            print('')


def get_duration_googleAPI(googleKey, malls):
    gmaps = googlemaps.Client(key=googleKey)
    while True:
        dt = get_loc_dt('Asia/Singapore')
        for mn0, (lat0, lon0) in malls.items():
            for mn1, (lat1, lon1) in malls.items():
                if mn0 == mn1:
                    continue
                res = gmaps.distance_matrix((lat0, lon0), (lat1, lon1), mode="driving", departure_time=dt)
                elements = res['rows'][0]['elements']
                dur = elements[0]['duration_in_traffic']['value']
                with open(TT_CSV, 'a') as w_csvfile:
                    writer = csv.writer(w_csvfile, lineterminator='\n')
                    writer.writerow([mn0, mn1, dur,
                                     dt.year, dt.month, dt.day, dt.weekday(), dt.hour, dt.minute])
        #
        time.sleep(MIN15SEC)


def summary_dayTrip():
    df = pd.read_csv(TT_CSV)
    df['timeslot'] = df.apply(lambda row: row['hour'] * N_TS_HOUR + int(row['minute'] / MIN15), axis=1)
    df['timeslot'] = df['timeslot'].astype(int)
    df['Date'] = df.apply(lambda row: date(row['year'], row['month'], row['day']), axis=1)
    dates = sorted(list(set(df['Date'])))
    for _date in dates:
        fpath = opath.join(sce_dpath, 'mTT-%d%02d%02d.csv' % (_date.year, _date.month, _date.day))
        if opath.exists(fpath):
            continue
        day_df = df[(df['Date'] == _date)]
        day_df.to_csv(fpath, index=False)


if __name__ == '__main__':
    dataCollection()
    # summary_dayTrip()

