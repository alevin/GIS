# QGIS notes: 

Data files should be downloaded from https://www.tidetech.org/data/san-francisco-sample/

for 5min interpolation run


```
cdo -intntime,6 san-francisco-sample.grb san-francisco-sample_5m.grb
```

```
qgis_sfbay_currents/data/san-francisco-sample_5m.nc
qgis_sfbay_currents/data/san-francisco-sample_5m.grb
qgis_sfbay_currents/data/san-francisco-sample.nc
qgis_sfbay_currents/data/san-francisco-sample.grb
```