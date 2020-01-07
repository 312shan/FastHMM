[中文版本的 README](README.md)
------------------------------

# FastHMM

A python package for HMM (Hidden Markov Model) model with fast train and decoding implementation

## Python version
test by using Python3

## Install
### pip
```bash
pip install FastHMM
```

### source
```bash
pip install git+https://github.com/312shan/FastHMM.git
```

## Usage
```python
from FastHMM.hmm import HMMModel

# test model training and predict
hmm_model = HMMModel()
hmm_model.train_one_line([("我", "r"), ("爱", "v"), ("北京", "ns"), ("天安门", "ns")])
hmm_model.train_one_line([("你", "r"), ("去", "v"), ("深圳", "ns")])
result = hmm_model.predict(["俺", "爱", "广州"])
print(result)

# test save and load model
hmm_model.save_model()
hmm_model = HMMModel().load_model()
result = hmm_model.predict(["我们", "爱", "深圳"])
print(result)
```

Output:
```python
[('俺', 'r'), ('爱', 'v'), ('广州', 'ns')]
[('我们', 'r'), ('爱', 'v'), ('深圳', 'ns')]
```

## Performance:
test on dataset 人民日报 
```
python .test/test_postagging.py
```
Output:
```text
train size 18484 ,test_size 1000
finish training
eval result: 
predict 57929 tags, 54228 correct,  accuracy 0.9361114467710473
runtime : 370.1029086 seconds
```
Most of time the consuming is on the decoding stage,
I tried many ways to implement viterbi algorithm,
The implementation I currently use is the fastest
If you have suggestions for improving this decoding algorithm, 
please let me know, thank you very much.

## Reference
[MicroHMM](https://github.com/howl-anderson/MicroHMM)   
[Hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model)    
[Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)  
