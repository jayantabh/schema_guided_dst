# ALBERT model

## Required packges
1. absl-py (for tests)
2. fuzzywuzzy
3. numpy
4. six
5. tensorflow

The checkpoint for pretrained ALBERT model and the dataset is already present in this directory

The files have been renamed to match with the BERT model in the baseline but they actually contain the ALBERT model except for the vocabulary file which is the same as the BERT model

### Training
Other flags defined in `train_and_predict.py` can be used to specify additional
options.

Inside the repo create a directory named data and place the dstc8-schema-guided-dialogue inside it and make the directories 
data/schema_embedding_dir, data/output_example_dir, data/output_ckpt_dir and run the following command

```shell
python -m schema_guided_dst.baseline.train_and_predict \
--bert_ckpt_dir schema_guided_dst/albert_base/ \
--dstc8_data_dir schema_guided_dst/data/dstc8-schema-guided-dialogue \
--schema_embedding_dir schema_guided_dst/data/schema_embedding_dir \
--dialogues_example_dir schema_guided_dst/data/output_example_dir \
--output_dir schema_guided_dst/data/output_ckpt_dir --dataset_split train \
--run_mode train --task_name dstc8_single_domain
```

### Prediction

The prediction step generates a directory with json files containing dialogues
in a format which is identical to the one used by the dataset. The generated
json files can be used by the evaluation script. Like training, prediction
comprises of similar three steps, all of which are done in
`train_and_predict.py`. The flag `--output_dir` must be same as the one used
during training and the prediction outputs are saved inside it as a
subdirectory `pred_res_{ckpt_num}` where `ckpt_num` is the global step
corresponding to the checkpoint used to create the predictions. Multiple
checkpoints can be specified for prediction as a comma separated list using the
flag `--eval_ckpt`.


```shell
! python -m schema_guided_dst.baseline.train_and_predict \
--bert_ckpt_dir schema_guided_dst/albert_base/ \
--dstc8_data_dir schema_guided_dst/data/dstc8-schema-guided-dialogue \
--schema_embedding_dir schema_guided_dst/data/schema_embedding_dir \
--dialogues_example_dir schema_guided_dst/data/output_example_dir \
--output_dir schema_guided_dst/data/output_ckpt_dir --dataset_split dev \
--run_mode predict --task_name dstc8_single_domain \
--eval_ckpt <model ckpt number>
```

## Evaluation

Evaluation is done using `evaluate.py` which calculates the values of different
metrics defined in `metrics.py` by comparing model outputs with ground truth.
The script `evaluate.py` requires that all model predictions should be saved in
one or more json files contained in a single directory (passed as flag
`prediction_dir`). The json files must have same format as the ground truth data
provided in the challenge repository. This script can be run using the following
command:


```shell
! python -m schema_guided_dst.evaluate \
--dstc8_data_dir schema_guided_dst/data/dstc8-schema-guided-dialogue \
--prediction_dir schema_guided_dst/data/output_ckpt_dir/pred_res_<model_ckpt_number>_dev_dstc8_single_domain_dstc8-schema-guided-dialogue/ --eval_set dev \
--output_metric_file schema_guided_dst/data/eval.json
```
