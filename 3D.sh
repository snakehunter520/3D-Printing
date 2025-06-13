# model_name=TimeLLM
# train_epochs=1
# learning_rate=0.01
# llama_layers=32

# batch_size=2
# d_model=16
# d_ff=32

# comment='TimeLLM-3D'

# python -u run_main_V1.py \
#   --task_name long_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/温度原因-1/ \
#   --data_path all.csv \
#   --model_id 3D_512_96 \
#   --model $model_name \
#   --data 3D \
#   --features MS \
#   --freq s \
#   --seq_len 256 \
#   --label_len 48 \
#   --pred_len 48 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --factor 3 \
#   --enc_in 21 \
#   --dec_in 21 \
#   --c_out 21 \
#   --d_model 32 \
#   --d_ff 32 \
#   --batch_size $batch_size \
#   --learning_rate $learning_rate \
#   --llm_layers $llama_layers \
#   --train_epochs $train_epochs \
#   --model_comment $comment

model_name=TimeLLM
train_epochs=1
learning_rate=0.01
llama_layers=32
num_classes=4

batch_size=16
d_model=16
d_ff=256

comment='TimeLLM-3D'

python -u run_main_V1.py \
  --task_name anomaly_classification \
  --is_training 1 \
  --root_path ./dataset/温度原因-1/ \
  --data_path all.csv \
  --model_id 3D_512_96 \
  --model $model_name \
  --data 3D \
  --features M \
  --freq s \
  --seq_len 256 \
  --label_len 48 \
  --pred_len 1 \
  --num_classes $num_classes \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 32 \
  --d_ff 32 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment





