TEMPERATURE=$1
NUM_SAMPLES=8
DEVICES=$7
REPO_AUTHOR=$2
LLM_NAME=$3
FEW_SHOT=$4
FLITER=$5
RANGE=$6
if [ "$RANGE" = "500" ]; then
    PROBLEM_FILE="downloads/math_splits/test.jsonl"
else
    PROBLEM_FILE="downloads/math_splits/test_filtered.jsonl"
fi
if [ "$FEW_SHOT" = "yes" ]; then
    EXTRA_ARG="--few_shot"
else
    EXTRA_ARG=""
fi

if [ "$FLITER" = "p" ]; then
    EXTRA_ARG2="--top_p 0.9"
elif [ "$FLITER" = "mp" ]; then
    EXTRA_ARG2="--min_p 0.05"
elif [ "$FLITER" = "n" ]; then
    EXTRA_ARG2=""
elif [ "$FLITER" = "tp" ]; then
    EXTRA_ARG2="--typical_sampling"
else
    FLITER="k"
    EXTRA_ARG2="--top_k 20"
fi

CUDA_VISIBLE_DEVICES=${DEVICES} python hugging_inference.py \
    --prompt_file ${PROBLEM_FILE} \
    --output_file results/${LLM_NAME}/${LLM_NAME}-${FLITER}-temp${TEMPERATURE}_${NUM_SAMPLES}.json \
    --max_new_tokens 1024 \
    --temperature ${TEMPERATURE} \
    --num_samples ${NUM_SAMPLES} \
    --batch_size 32 \
    --resume_generation \
    --checkpoint_path models/${REPO_AUTHOR}/${LLM_NAME}/ \
    --tokenizer_path models/${REPO_AUTHOR}/${LLM_NAME}/ \
    ${EXTRA_ARG} \
    ${EXTRA_ARG2} \
    --reparse \
    --infer_entropy

python cal_entropy.py \
    --answer_name ${LLM_NAME}-${FLITER}-temp${TEMPERATURE} \
    --problem_file ${PROBLEM_FILE} \
    --result_folder results/${LLM_NAME}/ \
    --num_samples ${NUM_SAMPLES}