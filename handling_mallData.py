import pickle
from geopy.distance import VincentyDistance
from shapely.geometry import Polygon, Point
#
from init_project import *
#
NORTH, EAST, SOUTH, WEST = 0, 90, 180, 270


def get_malls():
    ifpath = opath.join(dpath['geo'], 'mall-data.csv')
    ofpath = opath.join(dpath['home'], 'mallsInfo(%.2f).pkl' % ZONE_UNIT_KM)
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
            polygon_LatLon = [(mp.latitude, mp.longitude) for mp in moved_points]
            polygon_LatLon.append((moved_points[0].latitude, moved_points[0].longitude))
            malls[name] = (lat, lon, polygon_LatLon)
    with open(ofpath, 'wb') as fp:
        pickle.dump(malls, fp)
    return malls


def get_mallPoly():
    mall_polygons = []
    malls = get_malls()
    for mn, (_, _, polygon_LatLon) in malls.items():
        polygon_LonLat = [(LatLon[1], LatLon[0]) for LatLon in polygon_LatLon]
        mPoly = poly(polygon_LonLat)
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
