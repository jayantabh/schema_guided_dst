# Schema Guided Dialogue State Prediction (CS-7650 Natural Lanugage Processing Project)

This repository is based on the following repositories:
1. Schema Guided DST (https://github.com/google-research/google-research/tree/master/schema_guided_dst)
2. ALBERT (https://github.com/google-research/albert)

## Required packges
1. absl-py (for tests)
2. fuzzywuzzy
3. numpy
4. six
5. tensorflow


## Dataset

The dataset can be downloaded from the github repository for
[Schema Guided Dialogue State Tracking](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue)
challenge, which is a part of
[DSTC8](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).

### Required files

1. `cased_L-12_H-768_A-12`: pretrained [BERT-Base, Cased] model checkpoint.
   Download link at [bert repository](https://github.com/google-research/bert).

## Jupyter Notebook
The codes for setup, training, prediction and evaluation is given in schema_guided_dst.ipynb. The notebook was tested on Google Colab. Run all the cells in the notebook (except save and load model if those features are not required). Scripts can also be used for these tasks and the process is described below.

### Training

Training comprises of three steps, all of which are done in
`train_and_predict.py`:

1. **Creating TF record file containing training examples.** Examples are
   generated for each frame present in each user utterance. The output directory
   for the dataset is specified by the flag `--dialogues_example_dir`. If this
   directory already contains the generated examples, this step is skipped.
2. **Computing embeddings of schema elements.** Embeddings are generated for
   each intent, slot and categorical slot value for each service using a
   pre-trained model. The output directory for the schema embeddings is
   specified by the flag `--schema_embedding_dir`. If this directory already
   contains the generated schema embeddings, this step is skipped.
3. **Training the model.** Model parameters are optimized and the checkpoints
   are saved in directory specified by the flag `--output_dir`.

Other flags defined in `train_and_predict.py` can be used to specify additional
options.

```shell
python -m schema_guided_dst.baseline.train_and_predict \
--bert_ckpt_dir <downloaded_bert_ckpt_dir> \
--dstc8_data_dir <downloaded_dstc8_data_dir> \
--dialogues_example_dir <output_example_dir> \
--schema_embedding_dir <output_schema_embedding_dir> \
--output_dir <output_ckpt_dir> --dataset_split train --run_mode train \
--task_name dstc8_single_domain
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
python -m schema_guided_dst.baseline.train_and_predict \
--bert_ckpt_dir <downloaded_bert_ckpt_dir> \
--dstc8_data_dir <downloaded_dstc8_data_dir> \
--schema_embedding_dir <output_schema_embedding_dir> \
--dialogues_example_dir <output_example_dir> \
--output_dir <output_dir_for_trained_model> --dataset_split dev \
--run_mode predict --task_name dstc8_single_domain \
--eval_ckpt <comma_separated_ckpt_numbers>
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
python -m schema_guided_dst.evaluate \
--dstc8_data_dir <downloaded_dstc8_data_dir> \
--prediction_dir <generated_model_predictions> --eval_set dev \
--output_metric_file <path_to_json_for_report>
```
