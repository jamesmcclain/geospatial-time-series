# Unsupervised pretraining on Sentinel-2 imagery #

![image](https://github.com/jamesmcclain/geospatial-time-series/assets/11281373/c97af030-477b-42c7-a401-683e2d6a13c5)

## OSM Data ##

See [here](https://registry.opendata.aws/daylight-osm/) and [here](https://daylightmap.org/).

## Language Models ##

```python
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-xl')
model.max_seq_length = 4096
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"
embeddings = model.encode([[instruction,sentence]])
print(embeddings)
```
