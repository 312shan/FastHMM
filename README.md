[README written in English](README.en.md)
------------------------------

# FastHMM

实现快速训练和解码预测的 HMM（隐马尔可夫模型）的 python 包

## 版本依赖
Python3

## 安装
### pip
```bash
pip install FastHMM
```

### source
```bash
pip install git+https://github.com/312shan/FastHMM.git
```

## 用法示例
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

## 性能:
在人民日报数据集上进行测试
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
上面的 case 大部分时间消耗都在解码阶段，
我尝试了多种方法来实现维特比算法，
我当前使用的实现是当中最快的。
如果您有改进此解码算法的建议，
请让我知道，非常感谢。

## 参考
[Hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model)  
[Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)