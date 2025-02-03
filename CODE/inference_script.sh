TEMPERATURE=$1
NUM_SAMPLES=256
DEVICES=$7
LLM_REPO_AUTHOR=$2
LLM_NAME=$3
FEW_SHOT=$4
FLITER=$5
TASK=$6
LIMIT=100

if [ "$FLITER" = "p" ]; then
    EXTRA_ARG2="--top_p 0.9"
elif [ "$FLITER" = "n" ]; then
    EXTRA_ARG2=""
else
    FLITER="k"
    EXTRA_ARG2="--top_k 20"
fi

if [ "$FEW_SHOT" = "yes" ]; then
    EXTRA_ARG="--few_shot"
else
    EXTRA_ARG=""
fi

CUDA_VISIBLE_DEVICES=${DEVICES} python main.py \
  --model models/${LLM_REPO_AUTHOR}/${LLM_NAME} \
  --tasks ${TASK} \
  --limit ${LIMIT} \
  --max_length_generation 1024 \
  --temperature ${TEMPERATURE} \
  --do_sample True \
  --n_samples ${NUM_SAMPLES} \
  --batch_size 16 \
  --precision fp16 \
  --allow_code_execution \
  --save_generations \
  --save_generations_path results/${LLM_NAME}/${TASK}-${LIMIT}-temp${TEMPERATURE}-${NUM_SAMPLES}.json \
  --metric_output_path results/${LLM_NAME}/${TASK}-${LIMIT}-temp${TEMPERATURE}-${NUM_SAMPLES}-metrics.json \
  --save_every_k_tasks 1 \
  --load_generations_path results/${LLM_NAME}/${TASK}-${LIMIT}-temp${TEMPERATURE}-${NUM_SAMPLES}_${TASK}.json \
  ${EXTRA_ARG} \
  ${EXTRA_ARG2}
#  --load_generations_path results/${LLM_NAME}/${TASK}-${LIMIT}-temp${TEMPERATURE}-${NUM_SAMPLES}_${TASK}.json