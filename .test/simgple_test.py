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
