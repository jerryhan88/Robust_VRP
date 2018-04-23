import os.path as opath
import folium
import webbrowser
#
from handling_mallData import get_malls


html_fpath = 'viz_mallLocations.html'

#
malls = get_malls()

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

