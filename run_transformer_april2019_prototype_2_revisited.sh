#!/bin/bash
# A bash script for setting up the new problem, training the new model
#  and evaluating results
#  run with the following command
#  nohup ./run_transformer_april2019_prototype_2_revisited.sh > t2t_model_log.txt 2>&1 &
#  kill process with the following command (download the t2t_user and t2t_train directories after, save on d drive)
#  kill PID  (look at nvidia-smi for PID)


export TF_FORCE_GPU_ALLOW_GROWTH="true"


#  NOTE=og_transformer_temp_test101
#  NOTE=May2019Reboot_conv_transformer_tensorflow_test_101
#  NOTE=exp1_og_transformer_3
#NOTE=exp1_ctweqnumlayers_run_80
#NOTE=newtest_exp_debug_decoding_general_newSept_312
#NOTE=newtest_exp_debug_decoding_general_ogtrans_32k_bigsinglegpu_2048batch_354
#NOTE=newtest_exp_debug_decoding_general_newConvt_8k_tinysinglegpu_1024batch_366
#NOTE=newtest_exp_debug_decoding_general_newConvt_32k_base_16layers_2048batch_508

#NOTE=newtest_exp_debug_decoding_general_newConvt_32k_recreate_ogtrans_wcovt_516
#NOTE=newtest_exp_debug_alt_og_transcoder_convt_with_pointatten_causalpaddingonlyondecodermh_fixedMHcausal_645
#NOTE=keywhile_debug_evolve_87
NOTE=new_2020_loss_debug_2020_2


PROBLEM=translate_ende_wmt32k
# PROBLEM=translate_ende_wmt8k


#MODEL=transformer
#MODEL=transformer_original_april2019
#MODEL=conv_transformer_april2019
MODEL=transformer_original_april2019_evolve2

#HPARAMS=transformer_base_single_gpu
#HPARAMS=transformer_base_single_gpu_local
#HPARAMS=transformer_base_single_gpu_local_adjusted_batch
#HPARAMS=transformer_big_single_gpu_local
#HPARAMS=transformer_big_single_gpu_local_adjusted_batch
#HPARAMS=transformer_small
# DO NOT USE: HPARAMS=conv_transformer_small_wmtende_v3
#HPARAMS=conv_transformer_exp1_ctweqnumlayers1_evolve2
#HPARAMS=conv_transformer_exp1_ctweqnumlayers1_evolve6
#HPARAMS=conv_transformer_exp1_ctweqnumlayers1_evolve7
HPARAMS=conv_transformer_exp1_ctweqnumlayers1_evolve14


#HPARAMS=conv_transformer_exp1_ctweqnumlayers1
#HPARAMS=conv_transformer_exp1_ctweqnumparams2
#HPARAMS=transformer_big

#TRAIN_STEPS=50000
TRAIN_STEPS=10000
#TRAIN_STEPS=110000

#TRAIN_STEPS=150000
#TRAIN_STEPS=250000
#TRAIN_STEPS=175285
#TRAIN_STEPS=523000
#TRAIN_STEPS=132310

#TRAIN_STEPS=391145


TRAIN_DIR=$HOME/t2t_train/$PROBLEM/$MODEL-$HPARAMS-$NOTE
USR_DIR=$HOME/t2t_user_dir/DEFENSE_langage_model_experiements/Language_Model_April2019_Restart
DATA_DIR=$HOME/t2t_data
TMP_DIR=$HOME/t2t_datagen
#TMP_DIR=/tmp/t2t_datagen

mkdir -p $DATA_DIR $TMP_DIR $TRAIN_DIR
mkdir -p $TMP_DIR


# Generate data
t2t-datagen \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM






# Train
# *  If you run out of memory, add --hparams='batch_size=1024'.
t2t-trainer \
  --t2t_usr_dir=$USR_DIR \
  --data_dir=$DATA_DIR \
  --problem=$PROBLEM \
  --model=$MODEL \
  --hparams_set=$HPARAMS \
  --output_dir=$TRAIN_DIR \
  --train_steps=$TRAIN_STEPS \
  --allow_growth=True \
  --schedule=train \
  --save_checkpoints_secs=600
#batch_size=1500





#   TODO:  need to average the last few checkpoints here


#CP_NUM_1=147000
#CP_NUM_2=137000
#CP_NUM_3=127000
#CP_NUM_4=117000
#CP_NUM_5=107000
#python avg_checkpoints.py --checkpoints=$TRAIN_DIR/model.ckpt-$CP_NUM_1,$TRAIN_DIR/model.ckpt-$CP_NUM_2,$TRAIN_DIR/model.ckpt-$CP_NUM_3,$TRAIN_DIR/model.ckpt-$CP_NUM_4,$TRAIN_DIR/model.ckpt-$CP_NUM_5 --num_last_checkpoints=5  --output_path=$TRAIN_DIR/avg_models/$NOTE-last5.ckpt

#
t2t-avg-all --model_dir=$TRAIN_DIR --output_dir=$TRAIN_DIR/avg_models/$NOTE-last5.ckpt --n=4


#And evaluate the model using:
#t2t-decoder --data_dir=$DATA_DIR --problems=$PROBLEM --model=$MODEL --hparams_set=$HPARAMS --output_dir=$TRAIN_DIR --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" --decode_from_file=$DECODE_FILE --decode_to_file=translation.de
#t2t-bleu --translation=translation.de --reference=data/newstest2014.de




 # Decode


 DECODE_FILE=$DATA_DIR/newstest2014.en
 TRUE_TRANSLATION=$DATA_DIR/newstest2014.de
 PREDICTED_TRANSLATION=$DATA_DIR/translation.de
 #TRANSLATIONS_DIR=$DATA_DIR/model_checkpoint_translations

 BEAM_SIZE=4
 ALPHA=0.6

 t2t-decoder \
   --t2t_usr_dir=$USR_DIR \
   --data_dir=$DATA_DIR \
   --problem=$PROBLEM \
   --model=$MODEL \
   --hparams_set=$HPARAMS \
   --output_dir=$TRAIN_DIR \
   --decode_hparams="beam_size=$BEAM_SIZE,alpha=$ALPHA" \
   --decode_from_file=$DECODE_FILE \
   --decode_to_file=$PREDICTED_TRANSLATION \
   --tfdbg \
   --checkpoint_path=$TRAIN_DIR/avg_models/$NOTE-last5.ckpt/model.ckpt-$TRAIN_STEPS

 # See the translations
 # cat translation.en

 # Evaluate the BLEU score
 # Note: Report this BLEU score in papers, not the internal approx_bleu metric.
t2t-bleu --translation=$PREDICTED_TRANSLATION --reference=$TRUE_TRANSLATION

#MODEL_DIR=/t2t_train/translate_ende_wmt32k/transformer-transformer_base_single_gpu-exp1_og_transformer_3
#AVG_MODEL_DIR=/home/chris_r_fortuna/t2t_train/translate_ende_wmt32k/transformer-transformer_base_single_gpu-exp1_og_transformer_3/avg_model_checkpoint
#t2t-bleu  --model_dir=$MODEL_DIR  --data_dir=$DATA_DIR  --translations_dir=$TRANSLATIONS_DIR  --problems=$PROBLEM  --hparams_set=$HPARAMS  --source=$DECODE_FILE  --reference=$TRUE_TRANSLATION

#t2t-avg-all --model_dir=$MODEL_DIR --output_dir=$AVG_MODEL_DIR --n=5
