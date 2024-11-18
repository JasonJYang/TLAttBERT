#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import torch
import numpy as np
import pandas as pd
import data_loader.dms_finetune_dataset as module_data
from scipy.stats import pearsonr, spearmanr
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from parse_config import ConfigParser

    
def compute_metrics(eval_pred):
    predictions = torch.tensor(eval_pred.predictions)
    labels = torch.tensor(eval_pred.label_ids)

    mask = ~torch.isnan(labels)
    masked_labels = labels[mask]
    masked_predictions = predictions[mask]
    pearson_corr = pearsonr(masked_predictions, masked_labels)[0]
    spearman_corr = spearmanr(masked_predictions, masked_labels)[0]

    return {'pearson': pearson_corr, 'spearman': spearman_corr}

def get_predictions(trainer, test_dataset):
    idxs_test = test_dataset.idxs_final
    outcomes_test = test_dataset.outcomes_final
    cat_df_test = test_dataset.dms_df
    
    predictions = trainer.predict(test_dataset)
    mask = ~torch.isnan(idxs_test)

    masked_outcomes_test = outcomes_test[mask]
    masked_predictions = torch.tensor(predictions.predictions)[mask]
    masked_idxs_test = idxs_test[mask].int()

    predicted_df = pd.DataFrame({
        'predicted_outcome': masked_predictions.numpy()})
    predicted_df.index = masked_idxs_test.tolist()

    predicted_df = pd.merge(predicted_df, cat_df_test, left_index=True, right_index=True)
    return predicted_df

def get_inference_predictions(trainer, inference_dataset, task2id_dict):
    predictions = trainer.predict(inference_dataset).predictions
    print('The shape of predictions:', predictions.shape)
    # for each mutation, we will have a 1568-dim vector
    mutation_list = inference_dataset.mutation_df['mutation'].tolist()
    id2task_dict = {v: k for k, v in task2id_dict.items()}
    task_list = [id2task_dict[i] for i in range(len(task2id_dict))]
    prediction_result_df = pd.DataFrame(predictions, columns=task_list)
    prediction_result_df['mutation'] = mutation_list
    prediction_result_df = prediction_result_df[['mutation'] + task_list]
    return prediction_result_df

def main(config):
    logger = config.get_logger('train')

    # fix random seeds for reproducibility
    seed = config['dataset']['args']['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

    # dataset
    logger.info('Loading dataset...')
    config['dataset']['args']['logger'] = logger
    dataset = config.init_obj('dataset', module_data)
    train_dataset, valid_dataset, test_dataset = dataset.get_dataloader()
    n_targets = dataset.get_n_targets_for_LoRa()
    tokenizer = dataset.get_tokenizer()
    max_length = tokenizer.model_max_length
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id
    times = list(dataset.day2daystr_dict.keys())
    task2id_dict = dataset.task2id_dict
    
    # model
    logger.info('Loading model...')
    if config['model']['name'] == 'TLAttBERT':
        from model.AttBERT_model import load_AttBERT_model, AttBERTForRegression
        AttBERTModel = load_AttBERT_model(
            logger=logger, 
            max_length=max_length,
            vocab_size=vocab_size,
            pad_token_id=pad_token_id,
            model_args=config['model']['model_args'],
            times=times,
            time_embedding_type=config['model']['time_embedding_type'])
        logger.info('Loading model from checkpoint: {}'.format(config['resume']))
        checkpoint = torch.load(config['resume'], map_location='cpu')
        AttBERTModel.load_state_dict(checkpoint)
        model = AttBERTForRegression(AttBERT=AttBERTModel.bert, 
                                     n_targets=n_targets, 
                                     intermediate_dim=config['model']['args']['intermediate_dim'],
                                     dropout_rate=config['model']['args']['dropout'])
    else:
        raise ValueError(f"Unrecognized model: {config['model']['name']}")
    
    LoRa_config = LoraConfig(
        task_type="SEQ_CLS",
        r=8,
        lora_alpha=16,
        target_modules=["key", "query", "value", "dense"],
        lora_dropout=0.05,
        bias="lora_only",
        modules_to_save=["regressor"])
    LoRa_model = get_peft_model(model, LoRa_config)
    LoRa_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config.save_dir,
        evaluation_strategy="epoch",
        learning_rate=config['trainer']['lr'],
        per_device_train_batch_size=config['trainer']['batch_size'],
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=2,
        
        fp16=True,
        load_best_model_at_end=True,
        remove_unused_columns=False,
        logging_steps=config['trainer']['logging_steps'],
        num_train_epochs=config['trainer']['epochs'],
        weight_decay=config['trainer']['weight_decay'],
        
        logging_dir=config.log_dir,
        save_strategy="epoch",
        save_total_limit=1,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=LoRa_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        compute_metrics=compute_metrics,
    )

    logger.info('Start training...')
    trainer.train()

    logger.info('Save model...')
    trainer.save_model(config.save_dir)

    logger.info('Start prediction...')
    predicted_df = get_predictions(trainer, test_dataset)
    predicted_df.to_csv(str(config.save_dir) + '/predicted_dms_df.csv', index=False)

    logger.info('Start inference for future mutation...')
    inference_dataset = dataset.prepare_dataset_for_inference(
        inference_mutation_dir=config['inference']['inference_mutation_dir'],
        inference_aggregate_day_lineage=config['inference']['inference_aggregate_day_lineage'],
        ref_seq_name=config['inference']['ref_seq_name'])
    inference_predicted_df = get_inference_predictions(trainer, inference_dataset, task2id_dict)
    inference_predicted_df.to_csv(str(config.save_dir) + '/inference_predicted_dms_df.csv', index=False)
    
    logger.info('Finish training and inference!')

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default='config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-local_rank', '--local_rank', default=None, type=str,
                      help='local rank for nGPUs training')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)