# Double Probing For Text Entailment

# Setting
```
python3 -m install pip
pip install -r requirements.txt
```

# A usecase
```python
import os
from .dp_model.model import DPTransformer
from .dp_model.tokenizer import DPTokenizer

config = {}
model = DPTransformer("self-attention", config)
# model = DPTransformer("FTransform", config)

sentence1, sentence2 = "A self-care serves to breakdown human being limits", "the limits of human could be broken with self-attention"
scores = model.score(sentence1, sentence2)
print(scores)
```