CHECKPOINT_PATH=/l/users/amirbek.djanibekov/master-thesis/models/adapters/speechqformer_fairseq/checkpoints/hubertlarge/pretrain_tristage/checkpoint_best.pt
DATA_ROOT=/l/users/amirbek.djanibekov/master-thesis/models/adapters/speechqformer_fairseq/_data
SUBSET="fairseq_speech_test_librispeech|fairseq_text_test_librispeech"
USER_DIR=/l/users/amirbek.djanibekov/master-thesis/models/adapters/speechqformer_fairseq
BEAM=10
MAX_TOKENS=3000000

fairseq-generate ${DATA_ROOT} \
  --gen-subset ${SUBSET} \
  --user-dir ${USER_DIR} \
  --task qformer_speech \
  --arch speech_qformer_base \
  --path ${CHECKPOINT_PATH} \
  --max-tokens ${MAX_TOKENS} \
  --beam ${BEAM} \
  --scoring wer \
  --max-len-a 3 \
  --max-len-b 1000 \
  --sample-rate 16000 \
  --batch-size 1 