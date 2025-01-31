## TURN: Optimizing Temperature for Language Models with Multi-Sample Inference

This repository contains the code for the following paper:

**Weihua Du, Yiming Yang, Sean Welleck: "Optimizing Temperature for Language Models with Multi-Sample Inference"**

### Overview

### Setup

You can use our code for automatic temperature selection.  

We support models from Hugging Face or local checkpoints.

#### Step 1: Prepare the Test Data  

Ensure your test data is in JSONL format, structured as follows:  
```json
{"query": "[query 1]"}
{"query": "[query 2]"}
...
```

#### Step 2: Run the Inference Script  

Execute the following command to determine the optimal temperature:  
```bash
python predict.py --model_path [LLM_PATH] --data_path [DATA_PATH] --aggregation_strategy [MJ/BofN]
```
This will output a suitable temperature for multi-sample aggregation. We currently support majority voting (MJ) and Best-of-N (BofN).

> **TODO:** Extend support for additional aggregation strategies.

### Reproducing Results  

To reproduce the results in our paper, refer to the guidelines in:  
- `CODE/readme.md` for the MBPP dataset
- `MATH/readme.md` for the MATH dataset

### Acknowledgements  

Our project builds upon *Easy-to-Hard Generation* and *bigcode-evaluation-harness*.  
Special thanks to *vLLM* for its efficient inference infrastructure.  