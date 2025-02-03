# MATH Experiment

## Install
Run the setup script to install the necessary packages:
   ```bash
   python install -r requirements.txt
   ```
---

## Generate Samples

To generate samples, ensure that the models are placed under the `models/` directory. Use the following script:

```bash
./inference_script/inference_hugging_model.sh [Temperature] [HuggingFace Model User] [Model Name] [few-shot yes/no] [k/p/n] [num of problem] [device]
```

### Example
```bash
./inference_script/inference_hugging_model.sh 0.8 deepseek-ai deepseek-math-7b-instruct yes k 200 0
```

### Parameters
- **Temperature**: Controls the randomness of predictions by scaling the logits before applying softmax.
- **HuggingFace Model User**: The user or organization name on HuggingFace.
- **Model Name**: The name of the model to use.
- **few-shot yes/no**: Whether to use few-shot learning (`yes` or `no`).
- **k/p/n**: Sampling method(k: top-k=20, p: top-p=0.9, n: no regulation)
- **num of problem**: The number of problems to generate samples for.
- **device**: The device to run the model on (e.g., `0` for GPU).

---

## Inference Entropy

To calculate inference entropy, use the following script:

```bash
./inference_script/inference_entropy_all.sh [device] [HuggingFace Model User] [Model Name] [FEW_SHOT] [num of problem]
```

This script will compute the entropy of the model's predictions across each temperature level.

---

## Turning Point Analysis
For turning point analysis, put the temperature-entropy json file in a JSON dict named `data.json`, which should be in `[temperature]:[entropy]` format, for example:
```json
{
    "0.1": 0.01,
    "0.2": 0.03,
}
```

Then, run the command:
```bash
python ../turning_point.py
```

---

## Notes
- Place all models in the `models/` directory to avoid path issues.