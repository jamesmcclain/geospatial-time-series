#!/usr/bin/env python3

import argparse
import json

import boto3
import rasterio
import rasterio as rio
from pyproj import CRS
from rasterio.warp import transform


def list_red_files(bucket, prefix):
    s3 = boto3.client('s3')
    tif_files = []

    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('B04.tif'):
                tif_files.append(obj['Key'])

    return tif_files


def pixel_to_wgs84_rect(ds, pixel_bounds):
    """
    Convert a pixel-space rectangle to a WGS84 rectangle.

    Args:
    - ds: A rasterio dataset.
    - pixel_bounds: A tuple of (left, bottom, right, top) in pixel coordinates.

    Returns:
    - A tuple of (left, bottom, right, top) in WGS84 coordinates.
    """
    # Convert pixel bounds to dataset CRS using affine transformation
    left, top = ds.transform * (pixel_bounds[0], pixel_bounds[3])
    right, bottom = ds.transform * (pixel_bounds[2], pixel_bounds[1])

    # Convert the dataset CRS bounds to WGS84
    dst_crs = CRS.from_epsg(4326)  # WGS84
    xs = [left, right]
    ys = [top, bottom]
    lon, lat = transform(ds.crs, dst_crs, xs, ys)

    return (lon[0], lat[1], lon[1], lat[0])  # Reordering to match WGS84 format


def wgs84_to_pixel_rect(ds, wgs84_bounds):
    """
    Convert a WGS84 rectangle to a pixel-space rectangle.

    Args:
    - ds: A rasterio dataset.
    - wgs84_bounds: A tuple of (left, bottom, right, top) in WGS84 coordinates.

    Returns:
    - A tuple of (left, bottom, right, top) in pixel coordinates.
    """
    # Convert WGS84 bounds to dataset CRS
    dst_crs = CRS.from_epsg(4326)  # WGS84
    xs = [wgs84_bounds[0], wgs84_bounds[2]]
    ys = [wgs84_bounds[1], wgs84_bounds[3]]
    lon, lat = transform(dst_crs, ds.crs, xs, ys)

    # Convert dataset CRS bounds to pixel-space using the inverse affine transformation
    left_pixel, top_pixel = ~ds.transform * (lon[0], lat[1])
    right_pixel, bottom_pixel = ~ds.transform * (lon[1], lat[0])

    left_pixel = int(left_pixel)
    right_pixel = int(right_pixel)
    top_pixel = int(top_pixel)
    bottom_pixel = int(bottom_pixel)

    return (left_pixel, bottom_pixel, right_pixel, top_pixel)


def write_bbox_to_geojson(wgs84_bounds, filename):
    """
    Write a WGS84 bounding box to a GeoJSON file.

    Args:
    - wgs84_bounds: A tuple of (left, bottom, right, top) in WGS84 coordinates.
    - filename: Name of the GeoJSON file to write to.

    Returns:
    - None
    """
    # Create the GeoJSON structure
    geojson_data = {
        "type":
        "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type":
                "Polygon",
                "coordinates": [[
                    [wgs84_bounds[0], wgs84_bounds[1]],  # Bottom-left
                    [wgs84_bounds[0], wgs84_bounds[3]],  # Top-left
                    [wgs84_bounds[2], wgs84_bounds[3]],  # Top-right
                    [wgs84_bounds[2], wgs84_bounds[1]],  # Bottom-right
                    [wgs84_bounds[0], wgs84_bounds[1]]  # Close the loop
                ]]
            }
        }]
    }

    # Write to file
    with open(filename, 'w') as outfile:
        json.dump(geojson_data, outfile, indent=4)


if __name__ == "__main__":

    # yapf: disable
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument("cog", type=str)
    # parser.add_argument("geotiff", type=str)
    # yapf: enable

    args = parser.parse_args()

    # with rio.open(args.cog, 'r') as ds:
    #     # wgs84_box = pixel_to_wgs84_rect(ds, [0, 0, 512, 512])
    #     # pixel_box = wgs84_to_pixel_rect(ds, wgs84_box)
    #     # write_bbox_to_geojson(wgs84_box, args.geotiff)

    #     wgs84_box = pixel_to_wgs84_rect(ds, [7000, 4000, 7512, 4512])
    #     pixel_box = wgs84_to_pixel_rect(ds, wgs84_box)
    #     # write_bbox_to_geojson(wgs84_box, args.geotiff)
    #     print(pixel_box)

    bucket_name = 'sentinel-cogs'
    prefix = 'sentinel-s2-l2a-cogs/50/C/NP/2023/'

    print(list_red_files(bucket_name, prefix))
