[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1aJRtDxXyH-ux-O7dA_0MXkfG0MCghWqs?usp=sharing)
## Swin Transformer PyTorch Hub

This is just a quick way to load Swin Transformers from image classification from PyTorch Hub. This repository makes it possible to load Swin Transformers in 1 line of code. 

The official Swin transformer repository can be found here:

https://github.com/microsoft/Swin-Transformer

## Dependencies

- `torch` - PyTorch

## Load Model

```python
import torch
HUB_URL = "neuraforce/swin-transformer-hub"
MODEL_NAME = "swin_tiny_patch4_window7_224"
# check hubconf for more models.
model = torch.hub.load(HUB_URL, MODEL_NAME, pretrained=True) # load from torch hub
```

## Transforms

Transforms for passing in `PIL` images for inference.

```python
from torchvision import transforms as T
from PIL import Image

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

## Imagenet Labels

Get a list of imagenet labels.

```python
import json
from urllib.request import urlopen

URL = "https://raw.githubusercontent.com/neuraforce/swin-transformer-hub/main/imagenet_labels.json"
response = urlopen(URL)
classes = json.loads(response.read())
len(classes) # Should return 1000
```
