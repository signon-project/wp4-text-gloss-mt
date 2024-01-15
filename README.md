# A module for Text-Gloss translation

This repository implements the SignON_MT class, which enables the translation between spoken and sign languages represented by glosses. This contribution is part of the SignON Project (https://signon-project.eu) and, namely, is the implementation the multilingual model trained under the first approach reported in the deliverable _"Deliverable 4.8: Final Routines for transformation of text from and to InterL"_

## Languages and models

This version only inludes the model for multilingual translation for the following Sign Languages: Spanish Sign Language (LSE), German Sign Language (DGS), British Sign Language (BSL) and The Netherlands Sign Language (NGT).The model was trained using SLs corpora and the two learning approaches reported in  _"Deliverable 4.8: Final Routines for transformation of text from and to InterL"_.

<!---It includes models for Spanish Sign Language (LSE), German Sign Language (DGS), British Sign Language (BSL) and The Netherlands Sign Language (NGT) and it also implements a multilingual model. The models were trained using SLs corpora and the two learning approaches reported in  _"Deliverable 4.8: Final Routines for transformation of text from and to InterL"_.

| Language | Approach 1 link | Approach 2 link |
| ------------- | ------------- | ------------- |
| LSE | [LSE-1](https://github.com/google/sentencepiece) | [LSE-2](https://github.com/google/sentencepiece) |
| BSL | [BSL-1](https://github.com/google/sentencepiece) | [BSL-2](https://github.com/google/sentencepiece) |
| NGT | [NGT-1](https://github.com/google/sentencepiece) | [NGT-2](https://github.com/google/sentencepiece) |
| DGS | [DGS-1](https://github.com/google/sentencepiece) | [DGS-2](https://github.com/google/sentencepiece) |
--->

## Requirements
This research was developed using Python 3.8.0. Below, the library requirements are listed to assure the experiments reproducibility.

| Resource | Version/URL |
| ------------- | ------------- |
| Tensorflow | 2.4.1 |
| Transformers | 4.19.2 |
| Numpy | 1.19.5 |
| SentencePiece | [LINK](https://github.com/google/sentencepiece) |

## Example of usage
```python
model = SignON_MT()
spoken_input = 'tiefer luftdruck bestimmt in den nÃ¤chsten tagen unser wetter'
gloss_input = 'DRUCK TIEF KOMMEN'

encoded_sentence = model.encode(gloss_input.lower(), 'DGS')
decoded_sentence = model.decode(encoded_sentence, 'de_DE')
print(decoded_sentence)
```
#### Output:
    der tiefdruck zieht weiter nach norddeutschland
