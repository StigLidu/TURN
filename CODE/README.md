# MBPP Experiment

## Installation

We use the open-source project **[bigcode-evaluation-harness](https://github.com/bigcode-project/bigcode-evaluation-harness)** for testing coding tasks.

### Steps to Install:
Run the setup script to install the necessary packages:
   ```bash
   python install -e .
   ```
---

## Running Scripts

### Generate Samples
To generate samples, ensure your models are placed under the `models/` directory.

**Command Format:**
```bash
./test_script.sh [Temperature] [HuggingFace Model User] [Model Name] [few-shot yes/no] [k/p/n] [dataset] [device]
```

**Example:**
```bash
./test_script.sh 0.8 codellama CodeLlama-7b-Python-hf yes k mbpp 0
```

**Parameters:**
- **Temperature**: Sampling temperature for generation.
- **HuggingFace Model User**: Username or organization name on HuggingFace.
- **Model Name**: Name of the model to be tested.
- **few-shot yes/no**: Specify if few-shot learning should be used.
- **k/p/n**: Sampling method(k: top-k=20, p: top-p=0.9, n: no regulation)
- **dataset**: Dataset to be used (Currently only support MBPP).
- **device**: Device ID (e.g., `0` for the first GPU).

This command will generate 256 samples for the first 100 problems.

---

## Inference Entropy
To evaluate the inference entropy, run:
```bash
./entropy_script.sh [Temperature] [HuggingFace Model User] [Model Name] [few-shot yes/no] [k/p/n] [dataset] [device]
```

This command will generate 8 samples for the first 100 problems for entropy estimation.

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