import os.path as opath
import os
import csv
#
from init_project import dpath

TARGET_MALLS = ['IMM', 'Tampines Mall', '313']
TARGET_HOURS = list(range(7, 11))

ofpath = 'temp.csv'

with open(ofpath, 'w') as w_csvfile:
    writer = csv.writer(w_csvfile, lineterminator='\n')
    new_header = [
        'fromMall', 'toMall', 'duration',
        'year', 'month', 'day', 'dow', 'hour']
    writer.writerow(new_header)

for fn in sorted(os.listdir(dpath['mallTrip'])):
    if not fn.endswith('.csv'):
        continue
    with open(opath.join(dpath['mallTrip'], fn)) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            hour = int(row['hour'])
            if not (hour in TARGET_HOURS):
                continue
            fromMall, toMall = [row[cn] for cn in ['fromMall', 'toMall']]
            if fromMall == toMall:
                continue
            if fromMall in TARGET_MALLS and toMall in TARGET_MALLS:
                with open(ofpath, 'a') as w_csvfile:
                    writer = csv.writer(w_csvfile, lineterminator='\n')
                    writer.writerow([row[cn] for cn in ['fromMall', 'toMall', 'duration',
                                        'year', 'month', 'day', 'dow', 'hour']])
