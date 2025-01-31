### Run scripts

Generate samples (put models under `models/` first):
```bash
./inference_script/inference_hugging_model.sh [Temperature] [HuggingFace Model User] [Model Name] [few-shot yes/no] [k/p/n] [num of problem] [device]
```
Example:
```Bash
./inference_script/inference_hugging_model.sh 0.8 deepseek-ai deepseek-math-7b-instruct yes k 200 0
```

**Inference Entropy**: see `inference_script/inference_entropy_all.sh`

**Calculate Accuracy**: see `inference_script/cal_acc.sh`

**Turning Point**: see `turning_point.py`
