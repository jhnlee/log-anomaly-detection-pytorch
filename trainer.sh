#!/usr/bin/env bash

#required
VOCAB_PATH="./data/vocab.txt"
TRAIN_DATA_PATH="./data/train.pkl"
VAL_DATA_PATH="./data/val.pkl"
SAVE_PATH="./model_saved/"

#model params
EMBEDDING_HIDDEN_DIM=64
NUM_HIDDEN_LAYER=1
GRU_HIDDEN_DIM=256
DROPOUT_P=0.1
ATTENTION_METHOD="dot"

#train params
BATCH_SIZE=256
WARMUP_PERCENT=0.001
LEARNING_RATE=1e-5
EPOCHS=30
EVAL_STEP=100
GRAD_CLIP_NORM=1.0

#other parameters
NUM_WORKERS=8
DEVICE="cuda"
FP16=0
FP16_OPT_LEVEL="01"
SEED=0

#run traininer
python train.py\
	--embedding_hidden_dim=${EMBEDDING_HIDDEN_DIM}\
        --num_hidden_layer=${NUM_HIDDEN_LAYER}\
	--gru_hidden_dim=${GRU_HIDDEN_DIM}\
	--dropout_p=${DROPOUT_P}\
	--attention_method=${ATTENTION_METHOD}\
	--batch_size=${BATCH_SIZE}\
	--warmup_percent=${WARMUP_PERCENT}\
        --learning_rate=${LEARNING_RATE}\
	--epochs=${EPOCHS}\
	--eval_step=${EVAL_STEP}\
	--grad_clip_norm=${GRAD_CLIP_NORM}\
	--device=${DEVICE}\
	--fp16=${FP16}\
	--fp16_opt_level=${FP16_OPT_LEVEL}\
	--seed=${SEED}\
	--vocab_path=${VOCAB_PATH} \
	--train_data_path=${TRAIN_DATA_PATH} \
	--val_data_path=${VAL_DATA_PATH} \
	--save_path=${SAVE_PATH}

