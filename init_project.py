import os.path as opath
import sys, logging
import os
#
TAXI_HOME = opath.join(opath.join(opath.expanduser("~"), '..'), 'taxi')
ZONE_UNIT_KM = 0.1
#
dpath = {}
taxi_data_home = opath.join(opath.join(opath.dirname(opath.realpath(__file__)), '..'), 'taxi_data')
dpath['raw'] = opath.join(taxi_data_home, 'raw')
dpath['geo'] = opath.join(taxi_data_home, 'geo')
# --------------------------------------------------------------
dpath['home'] = opath.join(taxi_data_home, 'RobustVRP')
#
dpath['mallTrip'] = opath.join(dpath['home'], 'mallTrip(%.2f)' % ZONE_UNIT_KM)

for dn in ['home', 'mallTrip']:
    if not opath.exists(dpath[dn]):
        os.makedirs(dpath[dn])



def get_logger():
    py_fname = sys.argv[0][:-len('.py')]
    if '/' in py_fname:
        logger_name = '___log_%s' % py_fname.split('/')[-1]
    else:
        logger_name = '___log_%s' % py_fname
    logger = logging.getLogger('%s' % (logger_name))
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('%s.log' % (logger_name))
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger