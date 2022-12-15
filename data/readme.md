# Converting dependency parse to lexicalized constituency parse

The dependency parse is obtained from https://github.com/KhalilMrini/LAL-Parser/tree/master/data
The constituency parse is obtained from https://github.com/nikitakit/self-attentive-parser/tree/master/data

Install `nltk` first.


Usage:

1. Converting `conll` format dependency data to lexicalized tree into **bracketed string format lexicalized constituency parse**.

```bash
python dep2lex.py
```


2. Converting lexicalized tree to **dependency triples**

```bash
python lex2dep.py
```
