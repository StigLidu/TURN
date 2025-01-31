### Install

We use the opensource project bigcode_eval for testing code generation.

Run setup.py to install the necessary package.

### Run Scripts

Generate samples (put models under `models/` first):
```bash
./test_script.sh [Temperature] [HuggingFace Model User] [Model Name] [few-shot yes/no] [k/p/n] [dataset] [device]
```
For example:
```bash
./test_script.sh 0.8 codellama CodeLlama-7b-Python-hf yes k mbpp 0
```

**Inference Entropy**: see `entropy_script.sh`

**Turning Point**: see `../MATH/turning_point.py`
