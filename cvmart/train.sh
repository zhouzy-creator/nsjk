PIPELINE_CONFIG_PATH=/project/train/cvmart/ssd.config
MODEL_DIR=/project/train/cvmart/training
NUM_TRAIN_STEPS=10000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
export PYTHONPATH=$PYTHONPATH:/project/train/src_repo/tf-models/research:/project/train/src_repo/tf-models

  python model_main.py \
  --model_dir=$MODEL_DIR --num_train_steps=$NUM_TRAIN_STEPS \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --pipeline_config_path=$PIPELINE_CONFIG_PATH \
  --alsologtostderr
