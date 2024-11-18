# TLAttBERT
This is the implementation of our Temporal-Lineage Attention BERT (TLAttBERT) for SARS-CoV-2 mutation prediction.
# Software requirements
The entire codebase is written in python. Package requirements are as follows:
- python=3.11.4
- pytorch=2.0.1
- transformers=4.32.0
- peft=0.11.1
- biopython=1.81
- natsort=8.4.0
- scikit-learn=1.3.0
- scipy=1.11.1
- numpy=1.24.4
- pandas=1.5.3
- tqdm
# Data requirements
- Amino acid sequences of SARS-CoV-2 Spike protein is available at [GISAID](https://gisaid.org/).
- Deep mutational scanning dataset is available at [Bloom lab GitHub repository](https://github.com/jbloomlab/SARS2_RBD_Ab_escape_maps/blob/main/processed_data/escape_data_mutation.csv).
# How to use
## Pretrain
For the pretrain using amino acid squences of Spike protein, please use the following command
```bash
python hf_pretrain.py --config config/TLAttBERT/sample/TLAttBERT_sample_Cao_pretrain.json
```
## LoRa finetune and inference
For the LoRa finetune and inference using the DMS dataset, please modify the configuration file `config/TLAttBERT/sample/LoRa_Cao_dms_finetune.json` by replacing the `resume` of the pretrained model. Afterward, use the following command
```bash
python LoRa_finetune.py --config config/TLAttBERT/sample/LoRa_Cao_dms_finetune.json
```
# Licence
This project is available under the MIT license.
# Acknowledgement
We thank the original implementation of [Temporal Attention](https://github.com/guyrosin/temporal_attention).
# Contact
Jiannan Yang - jnyang@hku.hk 