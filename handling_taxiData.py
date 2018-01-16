import os.path as opath
import csv
#
from init_project import *
from handling_mallData import get_mallPoly


def run(yymm):
    ofpath = opath.join(dpath['mallTrip'], 'mallTrip-%s.csv' % yymm)
    if opath.exists(ofpath):
        print('The file had already been processed; %s' % yymm)
        return None
    with open(ofpath, 'wt') as w_csvfile:
        writer = csv.writer(w_csvfile, lineterminator='\n')
        new_header = [
            'fromMall', 'toMall', 'duration', 'distance',
            'departureTime', 'arrivalTime',
            'year', 'month', 'day', 'dow', 'hour',
            'vid', 'fare']
        writer.writerow(new_header)
    #
    mall_polygons = get_mallPoly()
    #
    yy, mm = yymm[:2], yymm[-2:]
    yyyy = '20%s' % yy
    year, month = map(int, [yyyy, mm])
    ifpath = opath.join(TAXI_HOME, '%s/%s/trips/trips-%s-normal.csv' % (yyyy, mm, yymm))
    with open(ifpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            startLon, startLat = map(eval, [row[cn] for cn in ['start-long', 'start-lat']])
            for poly in mall_polygons:
                if poly.is_including((startLon, startLat)):
                    departureLocation = poly.name
                    break
            else:
                continue
            endLon, endLat = map(eval, [row[cn] for cn in ['end-long', 'end-lat']])
            for poly in mall_polygons:
                if poly.is_including((endLon, endLat)):
                    arrivalLocation = poly.name
                    break
            else:
                continue
            #
            with open(ofpath, 'a') as w_csvfile:
                writer = csv.writer(w_csvfile, lineterminator='\n')
                new_row = [departureLocation, arrivalLocation]
                new_row += [row[cn] for cn in ['distance', 'duration']]
                new_row += [row[cn] for cn in ['start-time', 'end-time']]
                new_row += [year, month]
                new_row += [row[cn] for cn in ['start-day', 'start-dow', 'start-hour']]
                new_row += [row[cn] for cn in ['vehicle-id', 'fare']]
                writer.writerow(new_row)


if __name__ == '__main__':
    run('0901')
