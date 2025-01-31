DEVICES=$1
REPO_AUTHOR=$2
LLM_NAME=$3
FEW_SHOT=$4
FLITER=$5
STORAGE_DIR=""
RANGE=$6

mkdir -p TE_result/TE_result/$LLM_NAME/

bash inference_script/inference_entropy.sh 0.1 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES 
bash inference_script/inference_entropy.sh 0.2 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 0.3 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 0.4 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 0.5 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 0.6 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 0.7 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 0.8 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 0.9 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 1.0 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 1.1 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 1.2 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 1.3 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 1.4 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES
bash inference_script/inference_entropy.sh 1.5 $REPO_AUTHOR $LLM_NAME $FEW_SHOT $FLITER $RANGE $DEVICES