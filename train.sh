python train.py \
--student_number 00000000 \
--data_path /root/healthcare/ai612-project2-2023/ \
--model_path emilyalsentzer/Bio_ClinicalBERT \
--batch_size 1 \
--valid_percent 0.2 \
--log_interval 1000 \
--bert_unfreeze
# --device_id 1
