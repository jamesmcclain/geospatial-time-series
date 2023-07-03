#!/bin/zsh

for file in *.tif
do
    gdalwarp -tr 10 10 -r bilinear $file resampled_$file
done
gdal_merge.py -separate -o _cog.tif resampled_B01.tif resampled_B02.tif resampled_B03.tif resampled_B04.tif resampled_B05.tif resampled_B06.tif resampled_B07.tif resampled_B08.tif resampled_B8A.tif resampled_B09.tif resampled_B11.tif resampled_B12.tif
gdal_translate -co TILED=YES -co BLOCKXSIZE=512 -co BLOCKYSIZE=512 -co COMPRESS=DEFLATE _cog.tif cog.tif
gdaladdo --config COMPRESS_OVERVIEW DEFLATE --config INTERLEAVE_OVERVIEW PIXEL --config BLOCKXSIZE_OVERVIEW 512 --config BLOCKYSIZE_OVERVIEW 512 -r nearest cog.tif
rm _cog.tif resampled_*.tif
