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
尝试了多种方法来实现维特比算法，
当前使用的实现是其中最快的。
如果您有改建议，
欢迎留言，非常感谢。

## TODO
1. 增加一个 BMES 标注的字粒度的 FastHMM 序列标注用例脚本。（上面的 93% 是词粒度）
2. 增加基于 BMES 标注数据的 FastHMM 分词用例脚本。
3. 增加一个序列标注评估脚本，实现更多指标的自动评测。

## 参考
[MicroHMM](https://github.com/howl-anderson/MicroHMM)  
[Hidden Markov model](https://en.wikipedia.org/wiki/Hidden_Markov_model)    
[Viterbi algorithm](https://en.wikipedia.org/wiki/Viterbi_algorithm)  
