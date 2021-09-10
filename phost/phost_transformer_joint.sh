#!/bin/bash

#SBATCH --job-name=STtrans            # create a short name for your job
#SBATCH --output=transformer_asr-st-%A.out      # create a output file
#SBATCH --error=transformer_asr-st-%A.err       # create a error file
#SBATCH --partition=batch          # choose partition
#SBATCH --gpus=8              # gpu count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --nodes=1                  # node count
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32          # cpu-cores per task (>1 if multi-threaded tasks)

root_dir=/lustre/scratch/client/vinai/users/longdct
cd ${root_dir}

echo   Date              = $(date)
echo   Hostname          = $(hostname -s)
echo   Working Directory = $(pwd)
echo   Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES
echo   Number of Tasks Allocated      = $SLURM_NTASKS
echo   Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK

source miniconda3/bin/activate
conda activate iwslt21
cd ${root_dir}/test_augment/fairseq/phost

ST_SAVE_DIR=${root_dir}/st_transformer_ckpt
ASR_SAVE_DIR=${root_dir}/asr_transformer_ckpt
LOG_DIR=${root_dir}/test_augment/log
TENSORBOARD_DIR=${LOG_DIR}/tensorboard
DATADIR=${root_dir}/prep_data
CHECKPOINT_FILENAME=avg_last_10_checkpoint.pt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

LR=2e-3
MAX_UPDATE=1000

echo "Train ASR"
fairseq-train ${DATADIR} \
  --tensorboard-logdir ${TENSORBOARD_DIR} \
  --user-dir ./ \
  --config-yaml config_asr.yaml --train-subset train_asr --valid-subset dev_asr \
  --save-dir ${ASR_SAVE_DIR} --num-workers 20 --max-tokens 40000 --max-update ${MAX_UPDATE} \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr ${LR} --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 1 \
  --log-format simple --log-interval 100  --keep-last-epochs 10 \
  --skip-invalid-size-inputs-valid-test | tee ${LOG_DIR}/transformer_asr_augment.train.log

python scripts/average_checkpoints.py \
    --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 10 --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${DATADIR} \
  --config-yaml config_asr.yaml --gen-subset dev_asr --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct | tee ${LOG_DIR}/transformer_asr_augment.dev.log

fairseq-generate ${DATADIR} \
  --config-yaml config_asr.yaml --gen-subset test_asr --task speech_to_text \
  --path ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 50000 --beam 5 \
  --scoring wer --wer-tokenizer 13a --wer-lowercase --wer-remove-punct | tee ${LOG_DIR}/transformer_asr_augment.test.log

echo "Train ST"

fairseq-train ${DATADIR} \
  --tensorboard-logdir ${TENSORBOARD_DIR} \
  --user-dir ./ \
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${ST_SAVE_DIR} --num-workers 20 --max-tokens 40000 --max-update ${MAX_UPDATE} \
  --task speech_to_text --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr ${LR} --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 1 \
  --log-format simple --log-interval 100  --keep-last-epochs 10 \
  --skip-invalid-size-inputs-valid-test \
  --load-pretrained-encoder-from ${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME} | tee ${LOG_DIR}/transformer_st_augment.train.log

python scripts/average_checkpoints.py \
    --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${DATADIR} \
  --config-yaml config_st.yaml --gen-subset dev_st --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu | tee ${LOG_DIR}/transformer_st_augment.dev.log

fairseq-generate ${DATADIR} \
  --config-yaml config_st.yaml --gen-subset test_st --task speech_to_text \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu | tee ${LOG_DIR}/transformer_st_augment.test.log

