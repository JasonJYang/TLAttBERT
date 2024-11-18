# -*- coding: utf-8 -*-

import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted
from datasets import DatasetDict
from sklearn.model_selection import train_test_split
from data_loader.utility_tokenizer import AATimeLineageTokenizer


class MAADataset(object):
    def __init__(self, logger, 
                       data_dir, 
                       tokenizer_dir=None,
                       tokenizer_savedir=None, 
                       virus_name='rsv', 
                       cut_off_start='2020-01-01',
                       cut_off_end='2020-12-31',
                       seed=0,
                       test_ratio=0.2,
                       use_all_for_train=True,
                       no_mask_time=False,
                       no_mask_lineage=False,
                       dataset_name="VanillaBERTDataset",
                       sample_ratio=None,
                       sample_method=None,
                       use_cd_hit=False):
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.virus_name = virus_name
        self.seed = seed
        self.test_ratio = test_ratio
        self.use_all_for_train = use_all_for_train
        self.dataset_name = dataset_name
        self.tokenizer_savedir = tokenizer_savedir
        self.sample_ratio = sample_ratio
        self.sample_method = sample_method

        self.logger.info('Loading data...')
        from data_loader.utility_data import load_cov_data
        self.cut_off_start = pd.to_datetime(cut_off_start)
        self.cut_off_end = pd.to_datetime(cut_off_end)
        seqs_dict, _, self.ref_seq = load_cov_data(logger=logger, 
                                                    data_dir=self.data_dir, 
                                                    cut_off_start=self.cut_off_start,
                                                    cut_off_end=self.cut_off_end)

        self.day2daystr_dict, self.daystr2idx_dict = self._get_daystr2idx_dict()
        self.unique_daystr_list = list(self.daystr2idx_dict.keys())
        self.unique_lineage_list = self._get_lineage_list()

        self.seq2daystr_lineage_dict = self._process_seq_dict(seqs_dict)
        seq_list = [str(seq) for seq in self.seq2daystr_lineage_dict.keys()]

        if use_cd_hit:
            processed_seq_list = self._cd_hit_process(seq_list)
            seq_list = processed_seq_list
            self.seq2daystr_lineage_dict = {seq: self.seq2daystr_lineage_dict[seq] for seq in seq_list}
        
        self.max_len = max([len(seq) for seq in seq_list]) + 2
        self.logger.info('Max length: {}'.format(self.max_len))
        if dataset_name == 'TAttBERTDataset':
            self.max_len = self.max_len + 1
            self.logger.info('Max length for time prepended: {}'.format(self.max_len))

        if no_mask_time and no_mask_lineage:
            self.logger.info('No mask for time and lineage tokens...')
            no_mask_token = self.unique_daystr_list + self.unique_lineage_list
        elif no_mask_time and not no_mask_lineage:
            self.logger.info('No mask for time tokens...')
            no_mask_token = self.unique_daystr_list
        elif no_mask_lineage and not no_mask_time:
            self.logger.info('No mask for lineage tokens...')
            no_mask_token = self.unique_lineage_list
        else:
            no_mask_token = None

        if dataset_name == 'TLAttBERTDataset':
            self.logger.info('Using tokenizer with time and lineage prepended...')
            self.AAtokenizer = AATimeLineageTokenizer(logger, 
                                                      self.unique_daystr_list, 
                                                      self.unique_lineage_list, 
                                                      no_mask_token=no_mask_token)
        else:
            raise ValueError('Invalid dataset name.')

        self.tokenizer = self.AAtokenizer.get_tokenizer(max_len=self.max_len,
                                                        tokenizer_dir=tokenizer_dir,
                                                        tokenizer_savedir=tokenizer_savedir)
        self.logger.info('pad_token: {}; eos_token (sep_token): {}; bos_token (cls_token): {}'.format(
            self.tokenizer.pad_token, self.tokenizer.sep_token, self.tokenizer.cls_token))
        
    def get_days_list(self):
        return self.unique_daystr_list
        
    def get_token_list(self):
        return self.AAtokenizer.token_with_special_list

    def get_vocab_size(self):
        if self.dataset_name == 'ESM2Dataset':
            self.logger.info('Vocab size: {}'.format(len(self.tokenizer.get_vocab())))
            return len(self.tokenizer.get_vocab())
    
        self.logger.info('Vocab size: {}'.format(len(self.AAtokenizer.token2index_dict)))
        self.logger.info('Vocab size in tokenizer: {}'.format(len(self.tokenizer.get_vocab())))
        return len(self.AAtokenizer.token2index_dict)

    def get_pad_token_id(self):
        return self.tokenizer.pad_token_id

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_max_len(self):
        return self.max_len
    
    def get_ref_seq(self):
        return self.ref_seq
    
    def get_dataset(self):
        dataset_dict = self._create_dataset_dict()
        return dataset_dict
    
    def _get_daystr2idx_dict(self):
        self.logger.info('Getting all daystr2idx_dict...')
        day2daystr2idx_df = pd.read_csv(self.data_dir.joinpath('day2str2idx.csv'))
        day2daystr_dict = dict(zip(day2daystr2idx_df['day'], day2daystr2idx_df['daystr']))
        daystr2idx_dict = dict(zip(day2daystr2idx_df['daystr'], day2daystr2idx_df['idx']))
        return day2daystr_dict, daystr2idx_dict
    
    def _get_lineage_list(self):
        self.logger.info('Getting all lineage list...')
        lineage_df = pd.read_csv(self.data_dir.joinpath('lineage2firstdate.csv'))
        lineage_list = natsorted(list(set(lineage_df['lineage_formatted'])))
        self.logger.info('The number of lineages: {}'.format(len(lineage_list)))
        return lineage_list
    
    def _process_seq_dict(self, seqs_dict):
        self.logger.info('Processing sequence dictionary...')
        start_date = datetime.datetime(2019, 11, 17)
        self.logger.info('')
        
        seq2day_lineage_dict = {}
        unique_day_set, unique_lineage_set = set(), set()
        for seq, meta_list in tqdm(seqs_dict.items(), total=len(seqs_dict)):
            for meta in meta_list:
                date = meta['collection_date']
                lineage = meta['lineage']
                if date < start_date:
                    continue
                day = (date - start_date).days
                
                unique_day_set.add(day)
                unique_lineage_set.add(lineage)

                if seq not in seq2day_lineage_dict:
                    seq2day_lineage_dict[seq] = [tuple([day, lineage])]
                else:
                    seq2day_lineage_dict[seq].append(tuple([day, lineage]))

        self.logger.info('Unique days in data: {}'.format(len(unique_day_set)))
        self.logger.info('Unique lineages in data: {}'.format(len(unique_lineage_set)))

        seq2daystr_lineage_dict = {}
        for seq, day_lineage_list in seq2day_lineage_dict.items():
            seq2daystr_lineage_dict[seq] = [tuple([self.day2daystr_dict[day], lineage]) for day, lineage in sorted(day_lineage_list)]
        self.logger.info('The number of sequences: {}'.format(len(seq2daystr_lineage_dict)))
        
        return seq2daystr_lineage_dict
    
    def _cd_hit_process(self, seq_list, threshold=0.99):
        save_dir = Path(self.tokenizer_savedir)
        self.logger.info('Running CD-HIT using threshold {}...'.format(threshold))
        from pycdhit import cd_hit
        from Bio import SeqIO
        with open(save_dir.joinpath('seq.fasta'), 'w') as f:
            for i, seq in enumerate(seq_list):
                f.write('>seq{}\n'.format(i))
                f.write('{}\n'.format(seq))
        res = cd_hit(i=save_dir.joinpath('seq.fasta'), 
                     o=save_dir.joinpath('seq_cdhit'),
                     c=threshold)
        processed_seq_list = []
        for record in SeqIO.parse(save_dir.joinpath('seq_cdhit'), 'fasta'):
            processed_seq_list.append(str(record.seq))
        self.logger.info('Number of sequences after CD-HIT: {}'.format(len(processed_seq_list)))
        return processed_seq_list
    
    def _create_dataset_dict(self):
        seq_list = list(self.seq2daystr_lineage_dict.keys())
        seq_list.sort()
        train_seq_list, test_seq_list = train_test_split(seq_list, test_size=self.test_ratio, random_state=self.seed)
        if self.use_all_for_train:
            self.logger.info('Using all sequences for training...')
            train_seq_list = seq_list
        train_seq2daystr_lineage_dict = {seq: self.seq2daystr_lineage_dict[seq] for seq in train_seq_list}
        test_seq2daystr_lineage_dict = {seq: self.seq2daystr_lineage_dict[seq] for seq in test_seq_list}
        
        self.logger.info('Train seqs: {}, Test seqs: {}'.format(len(train_seq2daystr_lineage_dict), 
                                                                len(test_seq2daystr_lineage_dict)))

        train_dataset = self._create_dataset(train_seq2daystr_lineage_dict)
        test_dataset = self._create_dataset(test_seq2daystr_lineage_dict)

        dataset_dict = DatasetDict({'train': train_dataset, 'test': test_dataset})
        
        self.logger.info('Dataset')
        self.logger.info(dataset_dict)
        self.logger.info("Example of dataset")
        self.logger.info(dataset_dict["train"][0])

        return dataset_dict
    
    def _create_dataset(self, seq2daystr_lineage_dict):
        if self.dataset_name == "TLAttBERTDataset":
            from data_loader.utility_dataset import TLAttBERTDataset
            self.logger.info('Using attention dataset with time and lineage prepend...')
            dataset = TLAttBERTDataset(logger=self.logger,
                                       seq2daystr_lineage_dict=seq2daystr_lineage_dict,
                                       daystr2idx_dict=self.daystr2idx_dict,
                                       unique_daystr_list=self.unique_daystr_list,
                                       unique_lineage_list=self.unique_lineage_list,
                                       tokenizer=self.tokenizer,
                                       max_len=self.max_len,
                                       sample_ratio=self.sample_ratio,
                                       sample_method=self.sample_method)
            return dataset
        else:
            raise ValueError('Invalid dataset name.')