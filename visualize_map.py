import cartopy.crs as ccrs
import cartopy.io.shapereader as shapereader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from datetime import datetime, timedelta
from math import floor

# Get the names of the template detection files
templates = np.loadtxt('../Plourde_2015/templates_list.txt', \
    dtype={'names': ('name', 'family', 'lat', 'lon', 'depth', 'eH', \
    'eZ', 'nb'), \
         'formats': ('S13', 'S3', np.float, np.float, np.float, \
    np.float, np.float, np.int)}, \
    skiprows=1)

# Threshold for filtering the catalog
threshold = pd.read_csv('threshold_cc.txt', sep=r'\s{1,}', header=None, engine='python')
threshold.columns = ['family', 'threshold_FAME', 'threshold_perm']

# Beginning and end of the period we are looking at
tbegin = datetime(2007, 7, 1, 0, 0, 0)

# Name of catalog
catalog = 'catalog_2007_2009'

# Time step between two images (in days)
tstep = 5

# Time interval for counting LFEs (in days)
dt = 10

nmax = 0

for image in range(0, 146):
    latitude = np.zeros(np.shape(templates)[0])
    longitude = np.zeros(np.shape(templates)[0])
    number_LFEs = np.zeros(np.shape(templates)[0])

    t1 = tbegin + timedelta(days=image * tstep)
    t2 = t1 + timedelta(days=dt)

    # Loop on templates
    for i in range(0, np.shape(templates)[0]):
        # Get the time of LFE detections
        filename = 'catalogs/' + templates[i][0].astype(str)
        latitude[i] = templates[i][2]
        longitude[i] = templates[i][3]
        
        # Open LFE catalog
        namedir = 'catalogs/' + templates[i][0].astype(str)
        namefile = namedir + '/' + catalog + '.pkl'
        df = pickle.load(open(namefile, 'rb'))

        # Filter LFEs
        df = df.loc[df['cc'] * df['nchannel'] >= threshold['threshold_perm'].iloc[i]]

        # Keep only time interval
        df_date = pd.DataFrame({'year':df['year'], 'month':df['month'], 'day':df['day']})
        df_date = pd.to_datetime(df_date)
        df['date'] = df_date
        df_subset = df.loc[(df['date'] >= t1) & (df['date'] <= t2)]
        number_LFEs[i] = len(df_subset)

    df = pd.DataFrame(data={'longitude':longitude, 'latitude':latitude, 'number_LFEs':number_LFEs})

    CALIFORNIA_NORTH = 3311 # projection

    lonmin = -125.0
    lonmax = -121.0
    latmin = 39.0
    latmax = 42.0

    shapename = 'ocean'
    ocean_shp = shapereader.natural_earth(resolution='10m',
                                           category='physical',
                                           name=shapename)

    shapename = 'land'
    land_shp = shapereader.natural_earth(resolution='10m',
                                           category='physical',
                                           name=shapename)

    fig = plt.figure(figsize=(15, 15)) 
    ax = plt.axes(projection=ccrs.epsg(CALIFORNIA_NORTH))
    ax.set_extent([lonmin, lonmax, latmin, latmax], ccrs.Geodetic())
    ax.set_title('LFEs on {:04d}-{:02d}-{:02d}'.format(t1.year, t1.month, t1.day), fontsize=20)
    ax.gridlines(linestyle=":")

    for myfeature in shapereader.Reader(ocean_shp).geometries(): 
        ax.add_geometries([myfeature], ccrs.PlateCarree(), facecolor='#E0FFFF', edgecolor='black', alpha=0.5)
    for myfeature in shapereader.Reader(land_shp).geometries(): 
        ax.add_geometries([myfeature], ccrs.PlateCarree(), facecolor='#FFFFE0', edgecolor='black', alpha=0.5)

    cm = plt.cm.get_cmap('winter')
    sc = plt.scatter(df['longitude'], df['latitude'], c=df['number_LFEs'], \
        vmin=0, vmax=400, cmap=cm, s=20.0 * df['number_LFEs'], transform=ccrs.PlateCarree())
    cb = plt.colorbar(sc, orientation='horizontal', shrink=0.6, aspect=40)
    cb.set_label(label='Number of LFEs', size=20)
    cb.ax.tick_params(labelsize=20)

    plt.savefig('movie/LFE_{:03d}.png'.format(image + 1), format='png')
    plt.close(1)
