import os.path as opath
from functools import reduce
from datetime import datetime
from dateutil import tz
import pytz
import csv
import time
import googlemaps
#
from init_project import dpath

TARGET_MALLS = ['IMM', 'Tampines Mall', '313']
MIN30 = 30 * 60

get_loc_dt = lambda loc: datetime.utcnow().replace(tzinfo=pytz.utc).astimezone(tz.gettz(loc))


def run():
    ifpath = opath.join(dpath['geo'], 'mall-data.csv')
    malls = {}
    with open(ifpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            name = row['name']
            if name in TARGET_MALLS:
                lat, lon = map(float, [row[cn] for cn in ['latitude', 'longitude']])
                malls[name] = (lat, lon)

    ofpath = reduce(opath.join, [opath.expanduser('~'), 'Dropbox', 'Data', 'mallTravelTime_googleMaps.csv'])
    if not opath.exists(ofpath):
        with open(ofpath, 'w') as w_csvfile:
            writer = csv.writer(w_csvfile, lineterminator='\n')
            new_header = [
                'fromMall', 'toMall', 'duration',
                'year', 'month', 'day', 'dow', 'hour']
            writer.writerow(new_header)
    #
    googleKey = 'AIzaSyAQYLeLHyJvNVC7uIbHmnvf7x9XC6murmk'
    gmaps = googlemaps.Client(key=googleKey)
    while True:
        dt = get_loc_dt('Asia/Singapore')
        for mn0, (lat0, lon0) in malls.items():
            for mn1, (lat1, lon1) in malls.items():
                if mn0 == mn1:
                    continue
                res = gmaps.distance_matrix((lat0, lon0),
                                                        (lat1, lon1), mode="driving")
                elements = res['rows'][0]['elements']
                dur = elements[0]['duration']['value']
                with open(ofpath, 'a') as w_csvfile:
                    writer = csv.writer(w_csvfile, lineterminator='\n')
                    writer.writerow([mn0, mn1, dur,
                                     dt.year, dt.month, dt.day, dt.weekday(), dt.hour])
        #
        time.sleep(MIN30)


if __name__ == '__main__':
    run()

