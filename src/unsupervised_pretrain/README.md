# Unsupervised pretraining on Sentinel-2 imagery #

![image](https://github.com/jamesmcclain/geospatial-time-series/assets/11281373/c97af030-477b-42c7-a401-683e2d6a13c5)

## OSM Data ##

See [here](https://registry.opendata.aws/daylight-osm/) and [here](https://daylightmap.org/).

```python
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.wkt import loads

parquet_files = glob.glob('/datasets/relation/*')
dfs = []
for parquet_file in parquet_files:
    df = pd.read_parquet(parquet_file)
    dfs.append(df)

df = pd.concat(dfs)
df["geometry"] = df["wkt"].apply(loads)
gdf = gpd.GeoDataFrame(df)

bounding_box = Polygon([(-118.9500, 33.5500), (-118.1500, 33.5500), (-118.1500, 34.3500), (-118.9500, 34.3500)])            
intersects = gdf[gdf.intersects(bounding_box)]
intersects.length
```
