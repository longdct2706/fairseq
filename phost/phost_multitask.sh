#!/bin/bash

#SBATCH --job-name=STmulti            # create a short name for your job
#SBATCH --output=multitask-%A.out      # create a output file
#SBATCH --error=multitask-%A.err       # create a error file
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

echo $(pwd)

SAVE_DIR=${root_dir}/test_augment/multitask_ckpt
PRETRAINED_DIR=${root_dir}/test_augment/pretrained_multitask
LOG_DIR=${root_dir}/test_augment/log/multitask
TENSORBOARD_DIR=${LOG_DIR}/tensorboard
DATADIR=${root_dir}/test_augment/prep_data
CHECKPOINT_FILENAME=avg_last_5_checkpoint.pt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

mkdir -p $TENSORBOARD_DIR

LEARNING_RATE=5e-5
MAX_UPDATE=1000

fairseq-train ${DATADIR} \
  --user-dir ./ \
  --tensorboard-logdir ${TENSORBOARD_DIR} \
  --config-yaml ${PRETRAINED_DIR}/config.yaml \
  --train-subset train_st --valid-subset dev_st \
  --save-dir ${SAVE_DIR} \
  --num-workers 20 --ddp-backend no_c10d \
  --max-tokens 500000 --max-sentences 3 --max-tokens-valid 800000 --max-update ${MAX_UPDATE} \
  --task speech_text_joint_to_text \
  --criterion guided_label_smoothed_cross_entropy_with_accuracy \
  --label-smoothing 0.3 --guide-alpha 0.8 --disable-text-guide-update-num 5000 \
  --max-source-positions 800000 --enc-grad-mult 2.0 \
  --attentive-cost-regularization 0.02 \
  --arch dualinputxmtransformer_base \
  --w2v-path ${PRETRAINED_DIR}/wav2vec.pt \
  --load-pretrained-mbart-from ${PRETRAINED_DIR}/mbart.pt \
  --optimizer adam --lr ${LEARNING_RATE} --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 1.0 --seed 1 --update-freq 4\
  --log-format simple --log-interval 100  --keep-last-epochs 5 \
  --skip-invalid-size-inputs-valid-test \
  --skip-encoder-projection \
  --attention-dropout 0.3 --mbart-dropout 0.3 \
  --finetune-w2v-params all --finetune-mbart-decoder-params all \
  --finetune-mbart-encoder-params all --stack-w2v-mbart-encoder \
  --drop-w2v-layers 12 --normalize | tee ${LOG_DIR}/multitask.train.log

#python ../scripts/average_checkpoints.py \
#    --inputs ${ASR_SAVE_DIR} --num-epoch-checkpoints 5 --output "${ASR_SAVE_DIR}/${CHECKPOINT_FILENAME}"
cp ${SAVE_DIR}/checkpoint_last.pt ${SAVE_DIR}/${CHECKPOINT_FILENAME} 

fairseq-generate ${DATADIR} \
  --user-dir ./ \
  --config-yaml ${PRETRAINED_DIR}/config.yaml \
  --gen-subset dev_st --task speech_text_joint_to_text --load-speech-only \
  --skip-invalid-size-inputs-valid-test \
  --infer-target-lang vi \
  --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 800000 --beam 5 \
  --scoring sacrebleu | tee ${LOG_DIR}/multitask.dev.log

fairseq-generate ${DATADIR} \
  --user-dir ./ \
  --config-yaml ${PRETRAINED_DIR}/config.yaml \
  --gen-subset test_st --task speech_text_joint_to_text --load-speech-only \
  --skip-invalid-size-inputs-valid-test \
  --infer-target-lang vi \
  --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} --max-tokens 800000 --beam 5 \
  --scoring sacrebleu | tee ${LOG_DIR}/multitask.test.log
