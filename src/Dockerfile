FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN pip install rasterio==1.3.7 shapely==2.0.1 \
    accelerate==0.20.3 bitsandbytes==0.39.1 einops==0.6.1 transformers==4.30.2 \
    pytorch-metric-learning==2.1.2 \
    pandas==2.0.2 geopandas==0.13.2 pyarrow==12.0.1

WORKDIR /workdir

# docker build -f Dockerfile -t timeseries:2.0.1-cuda11.7-cudnn8-devel .
