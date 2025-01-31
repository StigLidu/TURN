# Usage
# For simgle file: bash inference_script/cal_acc.sh answer_name(without result_folder) problem_file result_folder yes/no
# For a specific temperature: bash inference_script/cal_acc.sh answer_name(without result_folder) problem_file result_folder yes/no

#ANSWER_NAME=$1
ADD_ARG_FLAG=$4
if [ "$ADD_ARG_FLAG" = "yes" ]; then
    EXTRA_ARG="--reward"
else
    EXTRA_ARG=""
fi

ANSWER_NAME=$1
PROBLEM_FILE=$2
RESULT_FOLDER=$3
EXTRA_ARG2="--resume"

python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 1 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG} ${EXTRA_ARG2}
python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 2 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG} ${EXTRA_ARG2}
python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 4 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG} ${EXTRA_ARG2}
python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 8 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG} ${EXTRA_ARG2}
python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 16 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG} ${EXTRA_ARG2}
python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 32 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG} ${EXTRA_ARG2}
python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 64 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG} ${EXTRA_ARG2}
python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 128 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG} ${EXTRA_ARG2}
python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 256 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG} ${EXTRA_ARG2}
python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 512 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG} ${EXTRA_ARG2}
python cal_acc.py --answer_name ${ANSWER_NAME} --max_samples 1024 --sample_times 50 --problem_file ${PROBLEM_FILE} --result_folder ${RESULT_FOLDER} ${EXTRA_ARG}  ${EXTRA_ARG2}