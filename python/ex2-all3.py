#!/usr/bin/env python

import netCDF4
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator,MaxNLocator,IndexLocator)
from matplotlib.dates import HourLocator, MonthLocator, YearLocator, AutoDateLocator, AutoDateFormatter, ConciseDateFormatter, IndexDateFormatter, DateFormatter
#import nc_time_axis
import cftime

import datetime as dt

import cartopy
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import seaborn as sns
import pandas as pd
#import earthpy as et

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Use white grid plot background from seaborn
#sns.set(font_scale=1.5, style="whitegrid")

np.warnings.filterwarnings("ignore")

file_location = "../qgis_sfbay_currents/data/san-francisco-sample.nc"
nc = netCDF4.Dataset(file_location)
nc.variables.keys()

lat = nc.variables["lat"][:]
lon = nc.variables["lon"][:]
time_var = nc.variables["time"]
dtime = netCDF4.num2date(time_var[:], time_var.units,only_use_cftime_datetimes=True)
htime=time_var[:]/60

start = dt.datetime(2016, 6, 30, 0, 0, 0)
stop = dt.datetime(2016, 7, 5, 0, 0, 0)

points = [
#    {"name": "Golden Gate", 	"lat": 37.819620, "lon": -122.478534},
    {"name": "SFbay enterance", "lat": 37.810, "lon": -122.502},
    {"name": "Eagles",   	"lat": 37.789,   "lon": -122.495  },
    {"name": "Deadman's", "lat": 37.790,   "lon": -122.500 },
#    {"name": "Buoy", "lat": 37.790,   "lon": -122.490},
    {"name": "Pancake", "lat": 37.7890614,   "lon": -122.4999884},
]

# -122.50199980,37.81059988
# -122.494601,37.788589
# -122.498537,37.789139

# find closest index to specified value
def near(array, value):
    idx = (abs(array - value)).argmin()
    return idx

def close_event():
    plt.close()

def wind_uv_to_dir(U,V):
    """
    Calculates the wind direction from the u and v component of wind.
    Takes into account the wind direction coordinates is different than the 
    trig unit circle coordinate. If the wind directin is 360 then returns zero
    (by %360)
    Inputs:
      U = west/east direction (wind from the west is positive, from the east is negative)
      V = south/noth direction (wind from the south is positive, from the north is negative)
    """
    WDIR= (270-np.rad2deg(np.arctan2(V,U)))%360
    return WDIR
    
def wind_uv_to_spd(U,V):
    """
    Calculates the wind speed from the u and v wind components
    Inputs:
      U = west/east direction (wind from the west is positive, from the east is negative)
      V = south/noth direction (wind from the south is positive, from the north is negative)
    """
    WSPD = np.sqrt(np.square(U)+np.square(V))
    return WSPD

# Create figure and plot space
#fig, ax2  = plt.subplots(figsize=(10,5))

#fig, (ax1,ax2,ax3,ax4) = plt.subplots(figsize=(10,5),dpi=300,nrows=4, sharex=True, subplot_kw=dict(frameon=False)) 
fig, (ax1,ax2,ax3) = plt.subplots(figsize=(15,15),dpi=300,nrows=3, sharex=True, subplot_kw=dict(frameon=False)) 

plt.subplots_adjust(hspace=1)

x_major_ticks = np.arange(0, 121, 5)
x_minor_ticks = np.arange(0, 121, 5)

ax1.set_xticks(x_major_ticks)
ax1.set_xticks(x_minor_ticks, minor=True)

color = 'tab:red'
ax1.set_xlabel("Hours")
ax1.set_ylabel('sea level(m)', color=color)
ax1.tick_params(axis='y', labelcolor=color)
ax1.minorticks_on()

color = 'tab:blue'
#ax1 = ax2.twinx()
ax2.set_ylabel("current\nmagnitude(m/s)")
ax2.tick_params(axis='y', labelcolor=color)
ax2.minorticks_on()

degree_sign = u'\N{DEGREE SIGN}'
color = 'tab:green'
ax3.set_ylabel("current\ndirection("+degree_sign+")")
ax3.tick_params(axis='y', labelcolor=color)
ax3.minorticks_on()

#color = 'tab:green'
#ax4.set_ylabel("current\ndirection")
#ax4.tick_params(axis='y', labelcolor=color)
#ax4.minorticks_on()

for p in points:
    lati = p["lat"]
    loni = p["lon"]
    pointname = p["name"]

    # Find nearest point to desired location (could also interpolate, but more work)
    ix = near(lon, loni)
    iy = near(lat, lati)


    istart = netCDF4.date2index(start, time_var, select="nearest")
    istop = netCDF4.date2index(stop, time_var, select="nearest")
    # print(istart,istop)

    vname = "var49"
    var = nc.variables["var82"]
    U2M = nc.variables["var49"]
    V2M = nc.variables["var50"]

    hs = var[istart:istop, iy, ix]
    US = U2M[istart:istop, iy, ix]
    VS = V2M[istart:istop, iy, ix]

    US_nans = US[:]
    VS_nans = VS[:]

    cs = np.sqrt(US_nans ** 2 + VS_nans ** 2)
    cd = wind_uv_to_dir(US_nans,VS_nans)

    tim = htime[istart:istop]

    # Create Pandas time series object
    ts = pd.Series(hs, index=tim, name=pointname)
    ts2 = pd.Series(cs, index=tim, name=pointname)
    ts3 = pd.Series(cd, index=tim, name=pointname)

    ts.plot(linestyle = '-',ax=ax1)
    #ts2.plot(linestyle = '-',ax=ax2)

    if pointname=='SFbay enterance':
      #ts4 = pd.Series(cd, index=tim, name=pointname)
      ts2.plot(linestyle = '-',ax=ax2)
      ts3.plot(linestyle = '-',ax=ax3)
    else:
      ts2.plot(linestyle = 'dashed',ax=ax2)
      ts3.plot(linestyle = 'dashed',ax=ax3)

    ts.to_csv('/tmp/'+pointname+'_tide.csv',header=True,index_label='Time')
    ts2.to_csv('/tmp/'+pointname+'_current_speed.csv',header=True,index_label='Time')
    ts3.to_csv('/tmp/'+pointname+'_current_dir.csv',header=True,index_label='Time')


fig.tight_layout()
ax2.tick_params(which='both', width=2)
ax2.tick_params(which='major', length=7)
ax1.grid(which="minor",axis='x', linestyle=':', linewidth='0.2', color='black')
ax1.grid(which="major",axis='x', linestyle=':', linewidth='0.4', color='black')
ax2.grid(which="minor",axis='x', linestyle=':', linewidth='0.2', color='black')
ax2.grid(which="major",axis='x', linestyle=':', linewidth='0.4', color='black')
ax3.grid(which="minor",axis='x', linestyle=':', linewidth='0.2', color='black')
ax3.grid(which="major",axis='x', linestyle=':', linewidth='0.4', color='black')

plt.xticks(rotation=90,fontsize = 8)
plt.subplots_adjust(bottom = 0.3)
"""
ax1.axis([0, 120, -2, 2])
ax2.axis([0, 120, 0, 2.5])
ax3.axis([0, 120, 0, 360])
"""
ax1.axis([46, 62, -2, 2])
ax2.axis([46, 62, 0, 2.5])
ax3.axis([46, 62, 0, 360])

ax3.set_ylim(0, 361)
#ax4.axis([0, 48, 0, 360])
#ax4.set_ylim(0, 361)
ax3.axhspan(45, 140, facecolor='green', alpha=0.3,label="ebb")
ax3.axhspan(220, 315, facecolor='blue', alpha=0.3,label="flood")
#ax.axhspan(9, 12, facecolor='red', alpha=0.5)
ax3.legend(bbox_to_anchor=(0.7, -0.2), prop={'size':10}, fancybox=True, shadow=True,ncol=6)

ax2.arrow( 0, 2.2, 1.0, 0, fc="blue", ec="blue", head_width=0.2, head_length=0.3 )
ax2.arrow( 7, 2.2, -1.0, 0, fc="green", ec="green", head_width=0.2, head_length=0.3 )
ax2.arrow( 13, 2.2, 1.0, 0, fc="blue", ec="blue", head_width=0.2, head_length=0.3 )
ax2.arrow( 20, 2.2, -1.0, 0, fc="green", ec="green", head_width=0.2, head_length=0.3 )
ax2.arrow( 25, 2.2, 1.0, 0, fc="blue", ec="blue", head_width=0.2, head_length=0.3 )
ax2.arrow( 32, 2.2, -1.0, 0, fc="green", ec="green", head_width=0.2, head_length=0.3 )
ax2.arrow( 38, 2.2, 1.0, 0, fc="blue", ec="blue", head_width=0.2, head_length=0.3 )
ax2.arrow( 45, 2.2, -1.0, 0, fc="green", ec="green", head_width=0.2, head_length=0.3 )
ax2.arrow( 50, 2.2, 1.0, 0, fc="blue", ec="blue", head_width=0.2, head_length=0.3 )
ax2.arrow( 56, 2.2, -1.0, 0, fc="green", ec="green", head_width=0.2, head_length=0.3 )
ax2.arrow( 63, 2.2, 1.0, 0, fc="blue", ec="blue", head_width=0.2, head_length=0.3 )
ax2.arrow( 70, 2.2, -1.0, 0, fc="green", ec="green", head_width=0.2, head_length=0.3 )
ax2.arrow( 75, 2.2, 1.0, 0, fc="blue", ec="blue", head_width=0.2, head_length=0.3 )
ax2.arrow( 81, 2.2, -1.0, 0, fc="green", ec="green", head_width=0.2, head_length=0.3 )
ax2.arrow( 88, 2.2, 1.0, 0, fc="blue", ec="blue", head_width=0.2, head_length=0.3 )
ax2.arrow( 94, 2.2, -1.0, 0, fc="green", ec="green", head_width=0.2, head_length=0.3 )
ax2.arrow( 88, 2.2, 1.0, 0, fc="blue", ec="blue", head_width=0.2, head_length=0.3 )
ax2.arrow( 94, 2.2, -1.0, 0, fc="green", ec="green", head_width=0.2, head_length=0.3 )
ax2.arrow( 100, 2.2, 1.0, 0, fc="blue", ec="blue", head_width=0.2, head_length=0.3 )
ax2.arrow( 106, 2.2, -1.0, 0, fc="green", ec="green", head_width=0.2, head_length=0.3 )

plt.savefig('/tmp/ex2-all3.png',bbox_inches='tight')
#plt.show()

