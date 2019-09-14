# Install
Cài đặt model spacy tiếng việt
```
1. sudo pip3 install spacy
2. sudo pip3 install https://github.com/trungtv/vi_spacy/raw/master/packages/vi_spacy_model-0.2.1/dist/vi_spacy_model-0.2.1.tar.gz --no-cache-dir > /dev/null && sudo python3 -m spacy link vi_spacy_model vi_spacy_model
```

Cài đặt model spacy fasttext
```
- Tải file .vec fasttext: https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.vi.300.vec.gz
- Convert fasttext thành thư viện python và link đến spacy

1. sudo python3 -m spacy init-model vi . --vectors-loc cc.vi.300.vec.gz
2. sudo python3 -m spacy package vocab vi -m meta.json
3. sudo python3 setup.py sdist
4. sudo pip3 install vi_model-0.0.0.tar.gz --no-cache-dir > /dev/null && sudo python3 -m spacy link vi_model vi_fasttext

```