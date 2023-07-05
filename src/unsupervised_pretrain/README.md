# Unsupervised pretraining on Sentinel-2 imagery #

![image](https://github.com/jamesmcclain/geospatial-time-series/assets/11281373/c97af030-477b-42c7-a401-683e2d6a13c5)

## OSM Data ##

See [here](https://registry.opendata.aws/daylight-osm/) and [here](https://daylightmap.org/).

## Language Models ##

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline

# model_name = "cerebras/Cerebras-GPT-111M"
model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16)
pipe = pipeline("feature-extraction", model=model, tokenizer=tokenizer)
features = pipe("Why hello there", return_tensors=True)

```
