FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

RUN --mount=type=cache,target=/root/.cache/pip pip install rasterio==1.3.7 shapely==2.0.1 \
    accelerate==0.20.3 bitsandbytes==0.39.1 einops==0.6.1 transformers==4.30.2 \
    pytorch-metric-learning==2.1.2 \
    pandas==2.0.2 geopandas==0.13.2 pyarrow==12.0.1

RUN --mount=type=cache,target=/root/.cache/pip pip install sentence-transformers==2.2.2 \
    InstructorEmbedding==1.0.1

WORKDIR /workdir

# DOCKER_BUILDKIT=1 docker build --build-arg BUILDKIT_INLINE_CACHE=1 -f Dockerfile -t timeseries:2.0.1-cuda11.7-cudnn8-devel .
