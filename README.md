# Double Probing For Text Entailment

# Setting
```
python3 -m install pip
pip install -r requirements.txt
```

# Train
```
python3 trainer.py "multinli_1.0_train.csv" "multinli_1.0_dev_matched.csv" 4 --max_epochs 10
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

[Universal Dependencies](https://universaldependencies.org/#language-u)