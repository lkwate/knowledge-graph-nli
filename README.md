# Enhance Natural Language Inference with Dependency Graph

The main goal of this project is to exploit information from Knowledge graphs (external as well as internal) to improve the language understanding of text. By blending the encoded representations of the dependency graph of the reference and the hypothesis in the Recognizing textual entailment (RTE) framework and the plain text encoding (with Roberta), we showed that structural knowledge is an effective source of information augmentation for natural language understanding.

The dependency tree is built here with the library [Spacy](https://spacy.io/usage/linguistic-features#dependency-parse) 

# Training and Evaluation
```sh
bash launch-training-graph.sh
```