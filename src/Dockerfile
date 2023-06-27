FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

RUN pip install rasterio==1.3.7 \
    pytorch-metric-learning==2.1.2 \
    pyarrow==12.0.1 \
    pandas==2.0.2 \
    geopandas==0.13.2 \
    shapely==2.0.1

WORKDIR /workdir

# docker build -f Dockerfile -t timeseries:1.13.1-cuda11.6-cudnn8-devel .
