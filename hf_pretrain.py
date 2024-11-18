#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
import torch
import numpy as np

from transformers import (
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer)

import data_loader.pretrain_dataset as module_data
from parse_config import ConfigParser


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
    config['dataset']['args']['tokenizer_savedir'] = config.save_dir
    config['dataset']['args']['tokenizer_dir'] = None
    dataset = config.init_obj('dataset', module_data)
    tokenized_dataset = dataset.get_dataset()
    tokenizer = dataset.get_tokenizer()
    days_list = dataset.get_days_list()
    max_len = dataset.get_max_len()
    vocab_size = dataset.get_vocab_size()
    pad_token_id = dataset.get_pad_token_id()
    
    # data collator
    if config['model']['name'] != 'GPT2':
        logger.info('Using MLM data collator...')
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=config['trainer']['mlm_probability'])
    else:
        logger.info('Using CLM data collator...')
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False)
    logger.info("Example of data collator with tokenized dataset:")
    out = data_collator([tokenized_dataset['train'][i] for i in range(5)])
    for key in out:
        logger.info(f'{key}: {out[key].shape}')
    
    # model
    logger.info('Loading model...')
    from model.AttBERT_model import load_AttBERT_model
    model = load_AttBERT_model(
        logger=logger, 
        max_length=max_len,
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        model_args=config['model']['args'],
        times=days_list,
        time_embedding_type=config['model']['time_embedding_type'])

    logger.info('Padding token id: {} in model'.format(model.config.pad_token_id))
    logger.info(model)
    trainable_params = model.parameters()
    params = sum([np.prod(p.size()) for p in trainable_params if p.requires_grad])
    logger.info(f'Trainable parameters {params}.')

    # training arguments
    training_args = TrainingArguments(
        output_dir=config.save_dir,
        overwrite_output_dir=True,
    
        per_device_train_batch_size=config['trainer']['batch_size'],
        per_device_eval_batch_size=config['trainer']['batch_size'],
        
        evaluation_strategy="epoch",
        logging_strategy="steps",
        logging_steps=config['trainer']['logging_steps'],

        gradient_accumulation_steps=8,
        num_train_epochs=config['trainer']['epochs'],

        weight_decay=config['trainer']['weight_decay'],
        learning_rate=config['trainer']['lr'],
        lr_scheduler_type="cosine",
        warmup_steps=config['trainer']['warmup_steps'],
        
        save_strategy="epoch",
        save_total_limit=1,
        dataloader_num_workers=8,
        load_best_model_at_end=True,

        no_cuda=False,  # Useful for debugging
        skip_memory_metrics=True,
        disable_tqdm=False,

        fp16=config['trainer']['fp16'], # Mixed precision training

        # report_to="none",
        logging_dir=config.log_dir)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],  # Defaults to None, see above
    )

    trainer.train()
    trainer.save_model(config.save_dir)


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