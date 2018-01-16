from os.path import expanduser
import os.path as opath
import os
#
TAXI_HOME = opath.join(opath.join(expanduser("~"), '..'), 'taxi')
#
dpath = {}
taxi_data_home = opath.join(opath.join(opath.dirname(opath.realpath(__file__)), '..'), 'taxi_data')
dpath['raw'] = opath.join(taxi_data_home, 'raw')
dpath['geo'] = opath.join(taxi_data_home, 'geo')
# --------------------------------------------------------------
dpath['home'] = opath.join(taxi_data_home, 'RobustVRP')
#
dpath['mallTrip'] = opath.join(dpath['home'], 'mallTrip')

for dn in ['home', 'mallTrip']:
    if not opath.exists(dpath[dn]):
        os.makedirs(dpath[dn])