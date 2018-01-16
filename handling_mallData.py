import os.path as opath
import csv
import pickle
from geopy.distance import VincentyDistance
from shapely.geometry import Polygon, Point
#
from init_project import *
ZONE_UNIT_KM = 0.1
NORTH, EAST, SOUTH, WEST = 0, 90, 180, 270


def get_malls():
    ifpath = opath.join(dpath['geo'], 'mall-data.csv')
    ofpath = opath.join(dpath['home'], 'mallsInfo.pkl')
    if opath.exists(ofpath):
        with open(ofpath, 'rb') as fp:
            malls = pickle.load(fp)
        return malls
    malls = {}
    mover = VincentyDistance(kilometers=ZONE_UNIT_KM)
    with open(ifpath) as r_csvfile:
        reader = csv.DictReader(r_csvfile)
        for row in reader:
            name = row['name']
            lat, lon = map(float, [row[cn] for cn in ['latitude', 'longitude']])
            p0 = [lat, lon]
            #
            moved_points = []
            for d1, d2 in [(WEST, NORTH), (EAST, NORTH),
                           (EAST, SOUTH), (WEST, SOUTH)]:
                mp = mover.destination(point=p0, bearing=d1)
                mp = mover.destination(point=mp, bearing=d2)
                moved_points.append(mp)
            #
            polygon = [(mp.latitude, mp.longitude) for mp in moved_points]
            polygon.append((moved_points[0].latitude, moved_points[0].longitude))
            malls[name] = (lat, lon, polygon)
    return malls


def get_mallPoly():
    mall_polygons = []
    malls = get_malls()
    for mn, (_, _, polygon) in malls.items():
        mPoly = poly(polygon)
        mPoly.name = mn
        mall_polygons.append(mPoly)
    return mall_polygons


class poly(Polygon):
    def __init__(self, poly_points):
        Polygon.__init__(self, poly_points)

    def is_including(self, coordinate):
        assert type(coordinate) == type(()), coordinate
        assert len(coordinate) == 2, len(coordinate)
        p = Point(*coordinate)
        return p.within(self)
