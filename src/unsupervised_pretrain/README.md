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

```python
from InstructorEmbedding import INSTRUCTOR
model = INSTRUCTOR('hkunlp/instructor-xl')
model.max_seq_length = 4096
sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
instruction = "Represent the Science title:"
embeddings = model.encode([[instruction,sentence]])
print(embeddings)
```

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

model = "tiiuae/falcon-7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
)
sequences = pipeline(
    "Why hello there",
    max_length=1024,
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
)
```
