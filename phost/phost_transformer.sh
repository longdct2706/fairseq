#!/bin/bash

#SBATCH --job-name=STtrans            # create a short name for your job
#SBATCH --output=phost_transformer_augment-%A.out      # create a output file
#SBATCH --error=phost_transformer_augment-%A.err       # create a error file
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

ST_SAVE_DIR=${root_dir}/test_augment/st_transformer_ckpt
DATADIR=${root_dir}/prep_data
CHECKPOINT_FILENAME=avg_10_last_ckpt.pt
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

DA_P_AUGM=0.8
DA_PITCH="-300,300"
DA_TEMPO="0.85,1.3"
DA_ECHO_DELAY="20,200"
DA_ECHO_DECAY="0.05,0.2"


fairseq-train ${DATADIR} \
  --user-dir ./ \
  --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
  --save-dir ${ST_SAVE_DIR} --num-workers 20 --max-tokens 40000 --max-update 100000 \
  --task speech_to_text_augment --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --report-accuracy \
  --arch s2t_transformer_s --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
  --warmup-updates 10000 --clip-norm 10.0 --seed 1 --update-freq 1 \
  --log-format simple --log-interval 100  --keep-last-epochs 10 \
  --skip-invalid-size-inputs-valid-test \
  --da-p-augm=${DA_P_AUGM} \
  --da-pitch=${DA_PITCH} \
  --da-tempo=${DA_TEMPO} \
  --da-echo-delay=${DA_ECHO_DELAY} \
  --da-echo-decay=${DA_ECHO_DECAY} \
  --normalize

python ../scripts/average_checkpoints.py \
    --inputs ${ST_SAVE_DIR} --num-epoch-checkpoints 10 --output "${ST_SAVE_DIR}/${CHECKPOINT_FILENAME}"

fairseq-generate ${DATADIR} \
  --user-dir ./ \
  --config-yaml config_st.yaml --gen-subset test_st --task speech_to_text_augment \
  --path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
  --max-tokens 50000 --beam 5 --scoring sacrebleu
