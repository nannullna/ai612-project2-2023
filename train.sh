python train.py \
--student_number 00000000 \
--data_path /data5/youngju/ai612-project2-2023/outcome/ \
--model_path emilyalsentzer/Bio_ClinicalBERT \
--batch_size 2 \
--valid_percent 0.2 \
--log_interval 5000 \
# --bert_unfreeze
--device_id 0