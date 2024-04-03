export PYTHONPATH="${PYTHONPATH}:/l/users/amirbek.djanibekov/master-thesis/models/adapters/speech_adapter"

DATA_ROOT=/l/users/amirbek.djanibekov/master-thesis/models/adapters/speechqformer_fairseq/_data
SAVE_DIR=/l/users/amirbek.djanibekov/master-thesis/models/adapters/speechqformer_fairseq/checkpoints_tokenized_train
# TRAIN_SET="fairseq_speech_train_clean_librispeech|fairseq_text_train_clean_librispeech"
# VALID_SET="fairseq_speech_dev_librispeech|fairseq_text_dev_librispeech"

# TRAIN_SET="speechtokenizer_codes_dev|fairseq_text_dev_librispeech"
# VALID_SET="speechtokenizer_codes_test|fairseq_text_test_librispeech"

TRAIN_SET="speechtokenizer_codes_train|fairseq_text_train_clean_librispeech"
VALID_SET="speechtokenizer_codes_dev|fairseq_text_dev_librispeech"

USER_DIR=/l/users/amirbek.djanibekov/master-thesis/models/adapters/speechqformer_fairseq

mkdir -p ${SAVE_DIR}

python -m torch.distributed.run --nproc_per_node=1 \
$(which fairseq-train) ${DATA_ROOT} \
  --save-dir ${SAVE_DIR} \
  --tensorboard-logdir ${SAVE_DIR} \
  --train-subset ${TRAIN_SET} \
  --valid-subset ${VALID_SET} \
  --distributed-world-size 1 \
  --distributed-port 0 \
  --ddp-backend pytorch_ddp \
  --nprocs-per-node 0 \
  --log-format json \
  --user-dir ${USER_DIR} \
  --seed 96 \
  --log-interval 100 \
  --amp \
  \
  --task qformer_speech_toknized \
  --sample-rate 16000 \
  --max-speech-sample-size 960000 \
  --min-speech-sample-size 16000 \
  \
  --num-workers 0 \
  --update-freq 2 \
  \
  --criterion speech_qformer \
  --optimizer adam \
  --adam-betas "(0.9, 0.98)" \
  --adam-eps 1e-05 \
  --weight-decay 0.05 \
  --clip-norm 20.0 \
  --lr 1e-4 \
  --lr-scheduler cosine \
  --warmup-init-lr 1e-6\
  --min-lr 1e-5 \
  --warmup-updates 1000 \
  \
  --max-update 8000000 \
  --batch-size 20 \
  --keep-last-epochs 10 \
  --save-interval-updates 1000 \
  --skip-invalid-size-inputs-valid-test \
  --required-batch-size-multiple 1 \
  \
  --arch speech_qformer_tokenized \
  --find-unused-parameters \
  --speechtokenizer-configpath /l/users/amirbek.djanibekov/master-thesis/models/SpeechTokenizer/huggingface/SpeechTokenizer/speechtokenizer_hubert_avg/config.json\
  --speechtokenizer-ckptpath /l/users/amirbek.djanibekov/master-thesis/models/SpeechTokenizer/huggingface/SpeechTokenizer/speechtokenizer_hubert_avg/SpeechTokenizer.pt \
  --qformer-dim 768 \
  --num-query-token 100 \
  --cross-attention-freq 2 \
  --wandb-project qformer-base-speechcodes-1gpu
