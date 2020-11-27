export BERT_TYPE="roberta_wwm"
#export BERT_TYPE="roberta_wwm_large"
# export BERT_DIR="../../models/$BERT_TYPE"
export BERT_DIR="/d/models/$BERT_TYPE"
export DATA_DIR="./data"
export OUTPUT_DIR="./out"
export CKPT_PATH=""
export MODE="test"
export GPU_IDS="0"


python test.py \
--gpu_ids=$GPU_IDS \
--mode=$MODE \
--bert_dir=$BERT_DIR \
--bert_type=$BERT_TYPE \
--raw_data_dir=$DATA_DIR \
--train_batch_size=4 \
