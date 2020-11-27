#export BERT_TYPE="roberta_wwm"
export BERT_TYPE="roberta_wwm_large"
# export BERT_DIR="../../models/$BERT_TYPE"
export BERT_DIR="/d/models/$BERT_TYPE"
export DATA_DIR="./data"
export OUTPUT_DIR="./out"
export CKPT_PATH="./out"
export MODE="train"
export GPU_IDS="0"


python train_combine.py \
--gpu_ids=$GPU_IDS \
--mode=$MODE \
--bert_dir=$BERT_DIR \
--output_dir=$OUTPUT_DIR \
--bert_type=$BERT_TYPE \
--raw_data_dir=$DATA_DIR \
--max_seq_len=150 \
--train_epochs=3 \
--train_batch_size=1 \
--lr=2e-5 \
--other_lr=2e-4 \
--eval_model
