BASE_DIR := $(shell pwd)


.PHONY: preprocess
preprocess: 
	python preprocess.py --config_name="base"


.PHONY: train
train: 
	python train.py --config_name="base"
		

.PHONY: eval
eval: 
	python eval.py --model_path="/path/to/model/checkpoint"