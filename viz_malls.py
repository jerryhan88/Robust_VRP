import os.path as opath
from geopy.distance import VincentyDistance
from shapely.geometry import Polygon, Point
import csv
import pickle
import folium
import webbrowser
#
from init_project import ZONE_UNIT_KM
from init_project import dpath
from mallTravelTime import TARGET_MALLS
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


def run():
    html_fpath = 'viz_mallLocations.html'

    #
    _malls = get_malls()
    malls = {}
    for mn, (lat, lon, polygon) in _malls.items():
        if mn not in TARGET_MALLS: continue
        malls[mn] = (lat, lon, polygon)

    max_lon, max_lat = -1e400, -1e400
    min_lon, min_lat = 1e400, 1e400
    for lat, lon, _ in malls.values():
        if max_lon < lon:
            max_lon = lon
        if lon < min_lon:
            min_lon = lon
        if max_lat < lat:
            max_lat = lat
        if lat < min_lat:
            min_lat = lat
    #
    lonC, latC = (max_lon + min_lon) / 2.0, (max_lat + min_lat) / 2.0
    map_osm = folium.Map(location=[latC, lonC], zoom_start=11)
    for mn, (lat, lon, polygon) in malls.items():
        map_osm.add_child(folium.PolyLine(locations=polygon, weight=1.0))
        folium.RegularPolygonMarker((lat, lon), popup=mn, fill_color='red', radius=6).add_to(map_osm)
    map_osm.save(html_fpath)
    #
    html_url = 'file://%s' % (opath.abspath(html_fpath))
    webbrowser.get('safari').open_new(html_url)


if __name__ == '__main__':
    run()