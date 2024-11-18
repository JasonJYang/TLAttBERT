# -*- coding: utf-8 -*-

import re
import datetime
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO
from pathlib import Path
from natsort import natsorted
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from data_loader.utility_tokenizer import AATimeLineageTokenizer


class DMSFinetuneDataset(object):
    def __init__(self, logger, 
                       data_dir, 
                       tokenizer_dir=None, 
                       virus_name='cov',
                       cut_off_start='2020-01-01',
                       cut_off_end='2020-12-31',
                       seed=0,
                       valid_ratio=0.1,
                       test_ratio=0.2,
                       batch_size=32,
                       dataset_name="VanillaBERTDataset",
                       use_data="spike_expression",
                       escape_study=None,
                       aggregate_day_lineage="mode",
                       for_LoRa=False,
                       target_virus='D614G',
                       target_condition='BD55-1241',
                       n_samples=10):
        self.logger = logger
        self.data_dir = Path(data_dir)
        self.virus_name = virus_name
        self.use_data = use_data
        self.seed = seed
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.escape_study = escape_study
        self.aggregate_day_lineage = aggregate_day_lineage
        self.for_LoRa = for_LoRa
        self.target_virus = target_virus
        self.target_condition = target_condition
        self.n_targets = None
        self.n_samples = n_samples
        self.tokenizer_dir = Path(tokenizer_dir)
        self.aa_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
                        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                        'Y', 'V', 'X', 'Z', 'J', 'U', 'B', 'Z']

        self.logger.info('Loading data...')
        from data_loader.utility_data import load_cov_data
        self.cut_off_start = pd.to_datetime(cut_off_start)
        self.cut_off_end = pd.to_datetime(cut_off_end)
        seqs_dict, _, _ = load_cov_data(logger=logger, 
                                        data_dir=self.data_dir, 
                                        cut_off_start=self.cut_off_start,
                                        cut_off_end=self.cut_off_end,
                                        get_ref_only=False)

        self.unique_day_list_train, self.unique_lineage_list_train = self.get_train_day_lineage(seqs_dict)

        self.logger.info('Getting reference sequence dataframe...')
        self.ref_seq_df = self._get_ref_seq_dataframe()

        if dataset_name == 'TLAttBERTDataset':
            self.logger.info('Using lineage prepend tokenizer...')
            self.AAtokenizer = AATimeLineageTokenizer(logger, 
                                                      unique_daystr_list=[], 
                                                      unique_lineage_list=[],
                                                      no_mask_token=None)
        else:
            raise ValueError('Invalid dataset name.')
        
        day2daystr2idx_df = pd.read_csv(self.data_dir.joinpath('day2str2idx.csv'))
        self.day2daystr_dict = dict(zip(day2daystr2idx_df['day'], day2daystr2idx_df['daystr']))
        self.daystr2idx_dict = dict(zip(day2daystr2idx_df['daystr'], day2daystr2idx_df['idx']))

        self.tokenizer = self.AAtokenizer.get_tokenizer(tokenizer_dir=tokenizer_dir)
        self.logger.info('pad_token: {}; eos_token (sep_token): {}; bos_token (cls_token): {}'.format(
            self.tokenizer.pad_token, self.tokenizer.sep_token, self.tokenizer.cls_token))
        
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_n_targets_for_LoRa(self):
        return self.n_targets
    
    def get_train_day_lineage(self, seq_dict):
        self.logger.info('Getting day lineage in the training data...')
        start_date = datetime.datetime(2019, 11, 17)
        unique_day_set, unique_lineage_set = set(), set()
        
        for seq, meta_list in tqdm(seq_dict.items(), total=len(seq_dict)):
            for meta in meta_list:
                date = meta['collection_date']
                lineage = meta['lineage']
                if date < start_date:
                    continue
                day = (date - start_date).days

                unique_day_set.add(day)
                unique_lineage_set.add(lineage)
        
        self.logger.info('The number of unique days is: {}'.format(len(unique_day_set)))
        self.logger.info('The number of unique lineages is: {}'.format(len(unique_lineage_set)))
        return list(unique_day_set), list(unique_lineage_set)

    def get_dataloader(self):
        dms_with_day_lineage_df = self._prepare_dms_data()
        self.logger.info('The shape of the dataframe of DMS with day lineage is: {}'.format(dms_with_day_lineage_df.shape))
        train_df, test_df = train_test_split(dms_with_day_lineage_df, test_size=self.test_ratio, random_state=self.seed)
        train_df, valid_df = train_test_split(train_df, test_size=self.valid_ratio * (1 - self.test_ratio), random_state=self.seed)
        self.logger.info('{} for training, {} for validation, and {} for testing.'.format(
            len(train_df), len(valid_df), len(test_df)))
        
        if self.for_LoRa:
            self.logger.info('Using LoRaDMSDataset...')
            from data_loader.utility_dataset import LoRaDMSDataset
            dataset_class = LoRaDMSDataset
            train_dataset = self._create_dataset(dataset_class, train_df)
            valid_dataset = self._create_dataset(dataset_class, valid_df)
            test_dataset = self._create_dataset(dataset_class, test_df)
            return train_dataset, valid_dataset, test_dataset
        
        train_dataset = self._create_dataset(dataset_class, train_df)
        valid_dataset = self._create_dataset(dataset_class, valid_df)
        test_dataset = self._create_dataset(dataset_class, test_df)

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_dataloader, valid_dataloader, test_dataloader
        
    def _prepare_dms_data(self):
        # dms data
        if self.use_data == 'spike_expression':
            dms_df = self._load_spike_expression_data()
        elif self.use_data == 'ace2_binding':
            dms_df = self._load_ace2_binding_data()
        elif self.use_data == 'escape':
            dms_df = self._load_escape_data()
        elif self.use_data == 'Cao_DMS':
            dms_df, self.task2id_dict, self.seq2id_dict = self._load_Cao_DMS_data()
        elif self.use_data == 'spike_ace2_DMS':
            dms_df, self.task2id_dict, self.seq2id_dict = self._load_spike_ace2_DMS_data()
        elif self.use_data == 'XBB15_Spike_ACE2_Binding':
            dms_df, self.task2id_dict, self.seq2id_dict = self._load_XBB15_Spike_ACE2_Escape_data()
        else:
            raise ValueError('Invalid data type.')
        
        self.logger.info('Removing the mutations with unknown amino acids...')
        dms_df = dms_df[~dms_df['mutation_name'].str.contains('\*')]
        self.logger.info('The shape of the dataframe of DMS data is: {}'.format(dms_df.shape))

        self.logger.info('Removing the mutations with incorrect amino acids considering ref seq...')
        for idx, row in dms_df.iterrows():
            mutation_name, site = row['mutation_name'], row['site']
            if self.ref_seq[site-1] != mutation_name[0]:
                dms_df = dms_df.drop(idx)
        self.logger.info('The shape of the dataframe of DMS data is: {}'.format(dms_df.shape))

        # mutation day lineage data
        mutation_day_lineage_df = self._get_mutation_day_lineage()
        mutation_day_lineage_df = mutation_day_lineage_df[mutation_day_lineage_df['mutation_name'].isin(dms_df['mutation_name'])]
        
        # aggregate the day lineage
        if self.aggregate_day_lineage == 'mode':
            self.logger.info('Aggregating the day lineage by mode...')
            day_mode = mutation_day_lineage_df['day'].mode()[0]
            lineage_mode = mutation_day_lineage_df['lineage'].mode()[0]
            mutation_day_lineage_df = mutation_day_lineage_df.groupby('mutation_name').agg({'day': lambda x: x.mode()[0], 
                                                                                            'lineage': lambda x: x.mode()[0]}).reset_index()
            dms_df = dms_df.merge(mutation_day_lineage_df, on='mutation_name', how='left')
            dms_df['day'] = dms_df['day'].fillna(day_mode)
            dms_df['lineage'] = dms_df['lineage'].fillna(lineage_mode)
        
        elif self.aggregate_day_lineage == 'first':
            self.logger.info('Aggregating the day lineage by first...')
            day_first = sorted(list(set(mutation_day_lineage_df['day'])))[0]
            lineage_first = sorted(list(set(mutation_day_lineage_df['lineage'])))[0]
            mutation_day_lineage_df = mutation_day_lineage_df.sort_values(by=['day', 'lineage'], ascending=True)
            mutation_day_lineage_df = mutation_day_lineage_df.groupby('mutation_name').agg({'day': 'first', 'lineage': 'first'}).reset_index()
            dms_df = dms_df.merge(mutation_day_lineage_df, on='mutation_name', how='left')
            dms_df['day'] = dms_df['day'].fillna(day_first)
            dms_df['lineage'] = dms_df['lineage'].fillna(lineage_first)

        elif self.aggregate_day_lineage == 'all':
            self.logger.info('Aggregating the day lineage by all...')
            day_mode = mutation_day_lineage_df['day'].mode()[0]
            lineage_mode = mutation_day_lineage_df['lineage'].mode()[0]
            dms_df = dms_df.merge(mutation_day_lineage_df, on='mutation_name', how='left')
            dms_df['day'] = dms_df['day'].fillna(day_mode)
            dms_df['lineage'] = dms_df['lineage'].fillna(lineage_mode)

        elif self.aggregate_day_lineage == 'last':
            self.logger.info('Aggregating the day lineage by last...')
            day_last = sorted(list(set(mutation_day_lineage_df['day'])))[-1]
            lineage_last = natsorted(list(set(mutation_day_lineage_df['lineage'])))[-1]
            
            # for each mutation in the escape data, get the day and lineage
            mutation_latest_day_lineage_dict = {'mutation_name': [], 'day': [], 'lineage': []}
            for group in mutation_day_lineage_df.groupby('mutation_name'):
                mutation_name, day_list, lineage_list = group[0], list(group[1]['day']), list(group[1]['lineage'])
                day = sorted(day_list)[-1]
                lineage = natsorted(set(lineage_list))[-1]
                mutation_latest_day_lineage_dict['mutation_name'].append(mutation_name)
                mutation_latest_day_lineage_dict['day'].append(day)
                mutation_latest_day_lineage_dict['lineage'].append(lineage)
            mutation_latest_day_lineage_df = pd.DataFrame(mutation_latest_day_lineage_dict)
            
            dms_df = dms_df.merge(mutation_latest_day_lineage_df, on='mutation_name', how='left')
            dms_df['day'] = dms_df['day'].fillna(day_last)
            dms_df['lineage'] = dms_df['lineage'].fillna(lineage_last)

        elif self.aggregate_day_lineage == None:
            self.logger.info('Not aggregating the day lineage...')

        else:
            raise ValueError('Invalid aggregation method.')
        
        dms_df['daystr'] = dms_df['day'].map(self.day2daystr_dict)
        self.logger.info('The shape of the dataframe of DMS data is: {}'.format(dms_df.shape))
        return dms_df
        
    def _create_dataset(self, dataset_class, data_df):
        if self.dataset_name == 'VanillaBERTDataset':
            self.logger.info('Creating VanillaBERTDataset...')
            dataset = dataset_class(logger=self.logger, 
                                    dms_df=data_df, 
                                    ref_seq=self.ref_seq,
                                    tokenizer=self.tokenizer, 
                                    max_len=self.tokenizer.model_max_length,
                                    prepend_time=False,
                                    prepend_lineage=False,
                                    time_attention=False,
                                    daystr2idx_dict=self.daystr2idx_dict,
                                    n_targets=self.n_targets,
                                    n_samples=self.n_samples)
            
        elif self.dataset_name == 'ESM2Dataset':
            self.logger.info('Creating ESM2Dataset...')
            dataset = dataset_class(logger=self.logger, 
                                    dms_df=data_df, 
                                    ref_seq=self.ref_seq,
                                    tokenizer=self.tokenizer, 
                                    max_len=1322,
                                    prepend_time=False,
                                    prepend_lineage=False,
                                    time_attention=False,
                                    daystr2idx_dict=self.daystr2idx_dict,
                                    n_targets=self.n_targets,
                                    n_samples=self.n_samples)

        elif self.dataset_name == 'TBERTDataset':
            self.logger.info('Creating TBERTDataset...')
            dataset = dataset_class(logger=self.logger, 
                                    dms_df=data_df, 
                                    ref_seq=self.ref_seq,
                                    tokenizer=self.tokenizer, 
                                    max_len=self.tokenizer.model_max_length,
                                    prepend_time=True,
                                    prepend_lineage=False,
                                    time_attention=False,
                                    daystr2idx_dict=self.daystr2idx_dict,
                                    n_targets=self.n_targets,
                                    n_samples=self.n_samples)
        
        elif self.dataset_name == 'AttBERTDataset':
            self.logger.info('Creating AttBERTDataset...')
            dataset = dataset_class(logger=self.logger, 
                                    dms_df=data_df, 
                                    ref_seq=self.ref_seq,
                                    tokenizer=self.tokenizer, 
                                    max_len=self.tokenizer.model_max_length,
                                    prepend_time=False,
                                    prepend_lineage=False,
                                    time_attention=True,
                                    daystr2idx_dict=self.daystr2idx_dict,
                                    n_targets=self.n_targets,
                                    n_samples=self.n_samples)
            
        elif self.dataset_name == 'TAttBERTDataset':
            self.logger.info('Creating TAttBERTDataset...')
            dataset = dataset_class(logger=self.logger, 
                                    dms_df=data_df, 
                                    ref_seq=self.ref_seq,
                                    tokenizer=self.tokenizer, 
                                    max_len=self.tokenizer.model_max_length,
                                    prepend_time=True,
                                    prepend_lineage=False,
                                    time_attention=True,
                                    daystr2idx_dict=self.daystr2idx_dict,
                                    n_targets=self.n_targets,
                                    n_samples=self.n_samples)
        
        elif self.dataset_name == 'TLAttBERTDataset':
            self.logger.info('Creating TLAttBERTDataset...')
            dataset = dataset_class(logger=self.logger, 
                                    dms_df=data_df, 
                                    ref_seq=self.ref_seq,
                                    tokenizer=self.tokenizer, 
                                    max_len=self.tokenizer.model_max_length,
                                    prepend_time=True,
                                    prepend_lineage=True,
                                    time_attention=True,
                                    daystr2idx_dict=self.daystr2idx_dict,
                                    n_targets=self.n_targets,
                                    n_samples=self.n_samples)

        return dataset

    def _get_ref_seq_dataframe(self):
        descriptions = []
        sequences = []
        for record in SeqIO.parse(self.data_dir.joinpath('nextclade.peptide.S_rename.fasta'), "fasta"):
            descriptions.append(record.description)
            sequences.append(str(record.seq).replace("*", ""))
        seq_df = pd.DataFrame({
            'target_virus': descriptions,
            'seq': sequences})
        return seq_df
        
    def _load_spike_expression_data(self):
        self.logger.info('Loading spike expression data...')
        
        dms_starr_2020_df = pd.read_csv(self.data_dir.parent.joinpath('DMS/Tyler_2020_Cell/1-s2.0-S0092867420310035-mmc2.csv'))
        self.logger.info('The shape of the dataframe of ACE2 binding and Spike expression is: {}'.format(dms_starr_2020_df.shape))
        self.logger.info('The sites are from {} to {}.'.format(min(dms_starr_2020_df['site_SARS2']), max(dms_starr_2020_df['site_SARS2'])))
        
        dms_starr_2020_df = dms_starr_2020_df[['mutation', 'site_SARS2', 'bind_avg', 'expr_avg']]
        dms_starr_2020_df['mutation_num'] = len(dms_starr_2020_df)
        dms_starr_2020_df = dms_starr_2020_df.rename(columns={'site_SARS2': 'site', 'bind_avg': 'ACE2 binding', 
                                                              'expr_avg': 'Spike expression', 'mutation': 'mutation_name'})
        
        dms_starr_spike_2020_df = dms_starr_2020_df[['mutation_name', 'site', 'Spike expression', 'mutation_num']]
        dms_starr_spike_2020_df.columns = ['mutation_name', 'site', 'target', 'mutation_num']
        dms_starr_spike_2020_df = dms_starr_spike_2020_df.dropna()
        self.logger.info('The shape of the dataframe of Spike expression after dropping NA is: {}'.format(dms_starr_spike_2020_df.shape))
        
        self.logger.info('Scaling Spike expression...')
        dms_starr_spike_2020_df.loc[:, "target"] = dms_starr_spike_2020_df["target"].transform(self.__scale_values)
        dms_starr_spike_2020_df.loc[:, "target"] = dms_starr_spike_2020_df['target'].clip(0, 1)

        dms_starr_spike_2020_df['study'] = '2020_Starr_Spike_Expression'
        self.logger.info('The shape of the dataframe of Spike expression is: {}'.format(dms_starr_spike_2020_df.shape))

        return dms_starr_spike_2020_df
    
    def _load_ace2_binding_data(self):
        self.logger.info('Loading ACE2 binding data...')
        dms_starr_2020_df = pd.read_csv(self.data_dir.parent.joinpath('DMS/Tyler_2020_Cell/1-s2.0-S0092867420310035-mmc2.csv'))
        self.logger.info('The shape of the dataframe of ACE2 binding and Spike expression is: {}'.format(dms_starr_2020_df.shape))
        self.logger.info('The sites are from {} to {}.'.format(min(dms_starr_2020_df['site_SARS2']), max(dms_starr_2020_df['site_SARS2'])))
        
        dms_starr_2020_df = dms_starr_2020_df[['mutation', 'site_SARS2', 'bind_avg', 'expr_avg']]
        dms_starr_2020_df['antibody_num'] = 0
        dms_starr_2020_df['mutation_num'] = len(dms_starr_2020_df)
        dms_starr_2020_df = dms_starr_2020_df.rename(columns={'site_SARS2': 'site', 'bind_avg': 'ACE2 binding', 
                                                              'expr_avg': 'Spike expression', 'mutation': 'mutation_name'})
        
        dms_starr_ace2_2020_df = dms_starr_2020_df[['mutation_name', 'site', 'ACE2 binding', 'antibody_num', 'mutation_num']]
        dms_starr_ace2_2020_df.columns = ['mutation_name', 'site', 'target', 'antibody_num', 'mutation_num']
        dms_starr_ace2_2020_df = dms_starr_ace2_2020_df.dropna()
        self.logger.info('The shape of the dataframe of ACE2 binding after dropping NA is: {}'.format(dms_starr_ace2_2020_df.shape))

        self.logger.info('Scaling ACE2 binding...')
        dms_starr_ace2_2020_df.loc[:, "target"] = dms_starr_ace2_2020_df["target"].transform(self.__scale_values)
        dms_starr_ace2_2020_df.loc[:, "target"] = dms_starr_ace2_2020_df['target'].clip(0, 1)

        dms_starr_ace2_2020_df['study'] = '2020_Starr_ACE2_Binding'
        self.logger.info('The shape of the dataframe of ACE2 binding is: {}'.format(dms_starr_ace2_2020_df.shape))

        return dms_starr_ace2_2020_df
    
    def _load_escape_data(self):
        self.logger.info('Loading escape data...')
        dms_df = pd.read_csv(self.data_dir.parent.joinpath('DMS/Allison_2022_Virus_Evolution/processed_data/escape_data.csv'))

        dms_df['mutation_name'] = dms_df['wildtype'] + dms_df['site'].astype(str) + dms_df['mutation']
        study_list = list(set(dms_df['study']))
        print('The studies are: {}'.format(study_list))
        if self.escape_study == None:
            self.logger.info('Using all escape data...')
        else:
            self.logger.info('Using escape data from {}...'.format(self.escape_study))
            if type(self.escape_study) != list:
                study_list = [self.escape_study]
            else:
                study_list = self.escape_study
                
        aggregated_df_list = []
        for study in study_list:
            study_df = dms_df[dms_df['study'] == study]
            antibody_num = len(set(study_df['condition']))
            grouped = study_df.groupby('mutation_name').agg({'mut_escape': 'mean', 'site': 'first'}).reset_index()
            grouped['study'] = study
            grouped['antibody_num'] = antibody_num
            grouped['mutation_num'] = len(grouped)
            aggregated_df_list.append(grouped)
        aggregated_df = pd.concat(aggregated_df_list)
        aggregated_df = aggregated_df[['mutation_name', 'site', 'mut_escape', 'study', 'antibody_num', 'mutation_num']]
        aggregated_df.columns = ['mutation_name', 'site', 'target', 'study', 'antibody_num', 'mutation_num']
        self.logger.info('The shape of the dataframe of escape data is: {}'.format(aggregated_df.shape))
        
        aggregated_df = aggregated_df.dropna(subset=['target'])
        self.logger.info('The shape of the dataframe of escape data after dropping NA is: {}'.format(aggregated_df.shape))

        return aggregated_df
    
    def _load_spike_ace2_DMS_data(self):
        def mutate_sequence(row):
            site = int(row["site"])
            seq = row["seq"]
            mut = row["mutant"]
            return seq[:site-1] + mut + seq[site:]
        def encode_categorical_variables(categories):
            unique_categories = sorted(set(categories))
            category_to_number = {category: i for i, category in enumerate(unique_categories)}
            numerical_categories = [category_to_number[category] for category in categories]
            return numerical_categories, category_to_number

        self.logger.info('Loading spike expression data...')
        dms_starr_2020_df = pd.read_csv(self.data_dir.parent.joinpath('DMS/Tyler_2020_Cell/1-s2.0-S0092867420310035-mmc2.csv'))
        self.logger.info('The shape of the dataframe of ACE2 binding and Spike expression is: {}'.format(dms_starr_2020_df.shape))
        self.logger.info('The sites are from {} to {}.'.format(min(dms_starr_2020_df['site_SARS2']), max(dms_starr_2020_df['site_SARS2'])))
        dms_starr_2020_df = dms_starr_2020_df[['mutation', 'site_SARS2', 'mutant', 'bind_avg', 'expr_avg']]
        dms_starr_2020_df = dms_starr_2020_df.rename(columns={'site_SARS2': 'site', 'bind_avg': 'ACE2 binding', 
                                                              'expr_avg': 'Spike expression', 'mutation': 'mutation_name'})
        dms_starr_spike_2020_df = dms_starr_2020_df[['mutation_name', 'site', 'mutant', 'Spike expression']]
        dms_starr_spike_2020_df.columns = ['mutation_name', 'site', 'mutant', 'target']
        dms_starr_spike_2020_df = dms_starr_spike_2020_df.dropna()
        self.logger.info('The shape of the dataframe of Spike expression after dropping NA is: {}'.format(dms_starr_spike_2020_df.shape))
        self.logger.info('Scaling Spike expression...')
        dms_starr_spike_2020_df.loc[:, "target"] = dms_starr_spike_2020_df["target"].transform(self.__scale_values)
        dms_starr_spike_2020_df.loc[:, "target"] = dms_starr_spike_2020_df['target'].clip(0, 1)
        dms_starr_spike_2020_df['group'] = '2020_Starr_Spike_Expression'

        self.logger.info('Loading ACE2 binding data...')
        dms_starr_2020_df = pd.read_csv(self.data_dir.parent.joinpath('DMS/Tyler_2020_Cell/1-s2.0-S0092867420310035-mmc2.csv'))
        self.logger.info('The shape of the dataframe of ACE2 binding and Spike expression is: {}'.format(dms_starr_2020_df.shape))
        self.logger.info('The sites are from {} to {}.'.format(min(dms_starr_2020_df['site_SARS2']), max(dms_starr_2020_df['site_SARS2'])))
        dms_starr_2020_df = dms_starr_2020_df[['mutation', 'site_SARS2', 'mutant', 'bind_avg', 'expr_avg']]
        dms_starr_2020_df = dms_starr_2020_df.rename(columns={'site_SARS2': 'site', 'bind_avg': 'ACE2 binding', 
                                                              'expr_avg': 'Spike expression', 'mutation': 'mutation_name'})
        dms_starr_ace2_2020_df = dms_starr_2020_df[['mutation_name', 'site', 'mutant', 'ACE2 binding']]
        dms_starr_ace2_2020_df.columns = ['mutation_name', 'site', 'mutant', 'target']
        dms_starr_ace2_2020_df = dms_starr_ace2_2020_df.dropna()
        self.logger.info('The shape of the dataframe of ACE2 binding after dropping NA is: {}'.format(dms_starr_ace2_2020_df.shape))
        self.logger.info('Scaling ACE2 binding...')
        dms_starr_ace2_2020_df.loc[:, "target"] = dms_starr_ace2_2020_df["target"].transform(self.__scale_values)
        dms_starr_ace2_2020_df.loc[:, "target"] = dms_starr_ace2_2020_df['target'].clip(0, 1)
        dms_starr_ace2_2020_df['group'] = '2020_Starr_ACE2_Binding'

        spike_ace2_df = pd.concat([dms_starr_spike_2020_df, dms_starr_ace2_2020_df])
        self.logger.info('The shape of the dataframe of Spike expression and ACE2 binding is: {}'.format(spike_ace2_df.shape))

        self.logger.info('Adding reference sequence for mutation...')
        self.ref_seq = self.ref_seq_df[self.ref_seq_df['target_virus'] == self.target_virus]['seq'].values[0]
        self.logger.info('Updating the reference sequence to {}...'.format(self.target_virus))
        spike_ace2_df['seq'] = self.ref_seq
        spike_ace2_df.loc[:, "seq"] = spike_ace2_df.apply(mutate_sequence, axis=1)

        task_ids, task2idx_dict = encode_categorical_variables(spike_ace2_df['group'].tolist())
        spike_ace2_df['task_id'] = task_ids
        self.n_targets = len(set(spike_ace2_df['task_id']))
        self.logger.info('{} tasks.'.format(len(task2idx_dict)))
        seq_ids, seq2id_dict = encode_categorical_variables(spike_ace2_df['seq'].tolist())
        spike_ace2_df['seq_id'] = seq_ids
        self.logger.info('{} sequences.'.format(len(seq2id_dict)))

        self.logger.info('The shape of the dataframe of spike and ace2 data is: {}'.format(spike_ace2_df.shape))

        return spike_ace2_df, task2idx_dict, seq2id_dict

    def _load_Cao_DMS_data(self):
        def get_mutation_name(row):
            site = int(row['site'])
            seq = row['seq']
            mut = row['mutation']
            return seq[site-1] + str(site) + mut
        def mutate_sequence(row):
            site = int(row["site"])
            seq = row["seq"]
            mut = row["mutation"]
            return seq[:site-1] + mut + seq[site:]
        def encode_categorical_variables(categories):
            unique_categories = sorted(set(categories))
            category_to_number = {category: i for i, category in enumerate(unique_categories)}
            numerical_categories = [category_to_number[category] for category in categories]
            return numerical_categories, category_to_number
        
        self.logger.info('Loading Cao DMS data...')
        dms_df = pd.read_csv(self.data_dir.parent.joinpath('DMS/Allison_2022_Virus_Evolution/processed_data/escape_data_mutation.csv'))
        # exclude 'SARS convalescents' and 'WT-engineered'
        dms_df = dms_df[~dms_df['source'].isin(['SARS convalescents', 'WT-engineered'])]
        # exclude IC50 >= 10
        dms_df = dms_df[dms_df['IC50'] < 10]
        self.logger.info('The shape of the dataframe of DMS data is: {}'.format(dms_df.shape))
        # only consider D614G virus
        self.logger.info('Only considering {} virus...'.format(self.target_virus))
        dms_df = dms_df[dms_df['target_virus'] == self.target_virus]
        self.logger.info('The shape of the dataframe of DMS data is: {}'.format(dms_df.shape))
        dms_df = dms_df[['condition', 'target_virus', 'IC50', 'neg_log_IC50', 'site', 'mutation', 'mut_escape', 'group']]
        
        self.logger.info('Scaling DMS data...')
        dms_df.loc[:,'group'] = dms_df["target_virus"] + "_" + dms_df["group"] #group = epitope group; group = task_group (used for balancing los weights)
        dms_df.loc[:,'data_group'] = dms_df["target_virus"] + "_" + dms_df["condition"] #condition = mAb type; data_group = task_name
        dms_df.loc[:,'mut_escape_w'] = dms_df["mut_escape"] * dms_df["neg_log_IC50"] #weighting by IC50
        dms_df.loc[:,"outcome_pre"] = np.log10(dms_df["mut_escape_w"]+1) # log transformation with pusedo count 1
        dms_df.loc[:,'outcome'] = dms_df.groupby('group')['outcome_pre'].transform(self.__scale_values) # scaling so that 0 and 95 percentile fell within the range 0–1
        dms_df.loc[:,'outcome'] = dms_df['outcome'].clip(0, 1) #clipping >1 value to 1

        self.logger.info('Adding reference sequence for mutation...')
        self.ref_seq = self.ref_seq_df[self.ref_seq_df['target_virus'] == self.target_virus]['seq'].values[0]
        self.logger.info('Updating the reference sequence to D614G...')
        antibody_escape_df = dms_df.merge(self.ref_seq_df, on='target_virus')
        antibody_escape_df['mutation_name'] = antibody_escape_df.apply(get_mutation_name, axis=1)
        antibody_escape_df.loc[:, "seq"] = antibody_escape_df.apply(mutate_sequence, axis=1)

        if self.target_condition is not None:
            self.logger.info('Only considering {}...'.format(self.target_condition))
            antibody_escape_df = antibody_escape_df[antibody_escape_df['condition'] == self.target_condition]
        else:
            self.logger.info('Considering all conditions...')
            self.logger.info('The number of conditions is: {}'.format(len(set(antibody_escape_df['condition']))))

        antibody_escape_df = antibody_escape_df[['seq', 'site', 'mutation_name', 'outcome', 'group', 'data_group']]
        antibody_escape_df = antibody_escape_df.rename(columns={'outcome': 'target'})

        task_ids, task2idx_dict = encode_categorical_variables(antibody_escape_df['data_group'].tolist())
        antibody_escape_df['task_id'] = task_ids
        self.n_targets = len(set(antibody_escape_df['task_id']))
        self.logger.info('{} tasks.'.format(len(task2idx_dict)))
        seq_ids, seq2id_dict = encode_categorical_variables(antibody_escape_df['seq'].tolist())
        antibody_escape_df['seq_id'] = seq_ids
        self.logger.info('{} sequences.'.format(len(seq2id_dict)))

        self.logger.info('The shape of the dataframe of escape data is: {}'.format(antibody_escape_df.shape))
        return antibody_escape_df, task2idx_dict, seq2id_dict

    def _load_BA2_Spike_ACE2_Binding_data(self):
        self.logger.info('Loading BA.2 Spike mediated entry data...')
        dms_df = pd.read_csv(self.data_dir.parent.joinpath('DMS/Dadonaite_2024_Nature/BA.2_Spike_ACE2_binding_summary.csv'))
        dms_df = dms_df[['site', 'wildtype', 'mutant', 'spike mediated entry', 'ACE2 binding']]
        self.logger.info('The shape of the dataframe of BA.2 Spike mediated entry and ACE2 binding data is: {}'.format(dms_df.shape))

    def _load_XBB15_Spike_ACE2_Escape_data(self):
        def get_mutation_name(row):
            site = int(row['site'])
            seq = row['seq']
            mut = row['mutant']
            return seq[site-1] + str(site) + mut
        def mutate_sequence(row):
            site = int(row["site"])
            seq = row["seq"]
            mut = row["mutant"]
            mutated_seq = seq[:site-1] + mut + seq[site:]
            mutated_seq = mutated_seq.replace('-', '')
            return mutated_seq
        def encode_categorical_variables(categories):
            unique_categories = sorted(set(categories))
            category_to_number = {category: i for i, category in enumerate(unique_categories)}
            numerical_categories = [category_to_number[category] for category in categories]
            return numerical_categories, category_to_number
        
        self.logger.info('Loading XBB15 Spike ACE2 and Escape...')
        dms_df = pd.read_csv(self.data_dir.parent.joinpath('DMS/Dadonaite_2024_Nature/XBB.1.5_Spike_DMS_summary.csv'))
        dms_df = dms_df[['site', 'wildtype', 'mutant', 'human sera escape', 'spike mediated entry', 'ACE2 binding']]
        dms_df['target_virus'] = 'XBB.1.5'
        # transfer the three columns to one column
        spike_dms_df = dms_df[['site', 'wildtype', 'mutant', 'target_virus', 'spike mediated entry']]
        spike_dms_df = spike_dms_df.rename(columns={'spike mediated entry': 'target'})
        spike_dms_df['condition'] = 'spike mediated entry'
        spike_dms_df = spike_dms_df.dropna()
        self.logger.info('The shape of the dataframe of XBB1.5 Spike mediated entry data is: {}'.format(spike_dms_df.shape))
        ace2_dms_df = dms_df[['site', 'wildtype', 'mutant', 'target_virus', 'ACE2 binding']]
        ace2_dms_df = ace2_dms_df.rename(columns={'ACE2 binding': 'target'})
        ace2_dms_df['condition'] = 'ACE2 binding'
        ace2_dms_df = ace2_dms_df.dropna()
        self.logger.info('The shape of the dataframe of XBB1.5 ACE2 binding data is: {}'.format(ace2_dms_df.shape))
        escape_dms_df = dms_df[['site', 'wildtype', 'mutant', 'target_virus', 'human sera escape']]
        escape_dms_df = escape_dms_df.rename(columns={'human sera escape': 'target'})
        escape_dms_df['condition'] = 'human sera escape'
        escape_dms_df = escape_dms_df.dropna()
        self.logger.info('The shape of the dataframe of XBB1.5 Escape data is: {}'.format(escape_dms_df.shape))
        dms_df = pd.concat([spike_dms_df, ace2_dms_df, escape_dms_df])
        self.logger.info('The shape of the dataframe of XBB1.5 Spike ACE2 and Escape data is: {}'.format(dms_df.shape))

        self.logger.info('Scaling DMS data...')
        dms_df.loc[:,'group'] = dms_df['target_virus'] + '_' + dms_df['condition']
        dms_df.loc[:,'data_group'] = dms_df['target_virus'] + '_' + dms_df['condition']
        dms_df.loc[:,'mut_escape_w'] = dms_df['target'] # no weighting
        dms_df.loc[:,"outcome_pre"] = dms_df["mut_escape_w"] # no transformation
        # dms_df.loc[:,"outcome_pre"] = np.log10(dms_df["mut_escape_w"]+1) # log transformation with pusedo count 1
        dms_df.loc[:,'outcome'] = dms_df.groupby('group')['outcome_pre'].transform(self.__scale_values) # scaling so that 0 and 95 percentile fell within the range 0–1]
        dms_df.loc[:,'outcome'] = dms_df['outcome'].clip(0, 1) #clipping >1 value to 1

        self.logger.info('Adding reference sequence for mutation...')
        self.ref_seq = self.ref_seq_df[self.ref_seq_df['target_virus'] == 'XBB.1.5']['seq'].values[0] # The ref_seq is aligned to Wuhan-Hu-1
        self.logger.info('Updating the reference sequence to XBB.1.5...')
        antibody_escape_df = dms_df.merge(self.ref_seq_df, on='target_virus')
        antibody_escape_df['mutation_name'] = antibody_escape_df.apply(get_mutation_name, axis=1)
        antibody_escape_df.loc[:, "seq"] = antibody_escape_df.apply(mutate_sequence, axis=1)

        if self.target_condition is not None:
            self.logger.info('Only considering {}...'.format(self.target_condition))
            antibody_escape_df = antibody_escape_df[antibody_escape_df['condition'] == self.target_condition]
        else:
            self.logger.info('Considering all conditions...')
            self.logger.info('The number of conditions is: {}'.format(len(set(antibody_escape_df['condition']))))
        
        antibody_escape_df = antibody_escape_df[['seq', 'site', 'mutation_name', 'outcome', 'group', 'data_group']]
        antibody_escape_df = antibody_escape_df.rename(columns={'outcome': 'target'})

        task_ids, task2idx_dict = encode_categorical_variables(antibody_escape_df['data_group'].tolist())
        antibody_escape_df['task_id'] = task_ids
        self.n_targets = len(set(antibody_escape_df['task_id']))
        self.logger.info('{} tasks.'.format(len(task2idx_dict)))
        seq_ids, seq2id_dict = encode_categorical_variables(antibody_escape_df['seq'].tolist())
        antibody_escape_df['seq_id'] = seq_ids
        self.logger.info('{} sequences.'.format(len(seq2id_dict)))

        self.logger.info('The shape of the dataframe of escape data is: {}'.format(antibody_escape_df.shape))
        return antibody_escape_df, task2idx_dict, seq2id_dict

    def __scale_values(self, x):
        max_value = np.percentile(x, 95)
        min_value = x.min()
        return (x - min_value) / (max_value - min_value)
    
    def _get_mutation_day_lineage(self):
        self.logger.info('Getting mutation day lineage...')
        mutation_day_lineage_df = pd.read_csv(self.data_dir.joinpath('mutation_info.csv'))
        self.logger.info('The shape of the dataframe of mutation day lineage is: {}'.format(mutation_day_lineage_df.shape))

        self.logger.info('Only keeping the mutations within training data...')
        mutation_day_lineage_df = mutation_day_lineage_df[mutation_day_lineage_df['day'].isin(self.unique_day_list_train)]
        mutation_day_lineage_df = mutation_day_lineage_df[mutation_day_lineage_df['lineage'].isin(self.unique_lineage_list_train)]
        self.logger.info('The shape of the dataframe of mutation day lineage is: {}'.format(mutation_day_lineage_df.shape))

        return mutation_day_lineage_df
    
    def prepare_dataset_for_inference(self, inference_mutation_dir, inference_aggregate_day_lineage, ref_seq_name):
        self.logger.info('Preparing dataset for inference...')
        # get reference seq
        ref_seq = self.ref_seq_df[self.ref_seq_df['target_virus'] == ref_seq_name]['seq'].values[0]
        ref_seq = ref_seq.replace('-', '')
        self.logger.info('The reference sequence of {} is: {}'.format(ref_seq_name, ref_seq))

        def mutate_sequence(row):
            site = row['site']
            mut = row["mutation"]
            mutated_seq = ref_seq[:site-1] + mut + ref_seq[site:]
            mutated_seq = mutated_seq.replace('-', '')
            return mutated_seq
        # a dataframe: seq, site, mutation_name
        mutation_df = pd.read_csv(inference_mutation_dir)
        # get mutated seq
        mutation_df['site'] = mutation_df.apply(lambda x: int(re.findall(r'[0-9]+', x['mutation'])[0]), axis=1)
        mutation_df.loc[:, "seq"] = mutation_df.apply(mutate_sequence, axis=1)
        # get day lineage info
        mutation_day_lineage_df = self._get_mutation_day_lineage()
        mutation_day_lineage_df = mutation_day_lineage_df.rename(columns={'mutation_name': 'mutation'})
        mutation_day_lineage_df = mutation_day_lineage_df[mutation_day_lineage_df['mutation'].isin(mutation_df['mutation'])]

        # aggregate the day lineage
        if inference_aggregate_day_lineage == 'mode':
            self.logger.info('Aggregating the day lineage by mode...')
            day_mode = mutation_day_lineage_df['day'].mode()[0]
            lineage_mode = mutation_day_lineage_df['lineage'].mode()[0]
            mutation_day_lineage_df = mutation_day_lineage_df.groupby('mutation').agg({'day': lambda x: x.mode()[0], 
                                                                                       'lineage': lambda x: x.mode()[0]}).reset_index()
            mutation_df = mutation_df.merge(mutation_day_lineage_df, on='mutation', how='left')
            mutation_df['day'] = mutation_df['day'].fillna(day_mode)
            mutation_df['lineage'] = mutation_df['lineage'].fillna(lineage_mode)
        
        elif inference_aggregate_day_lineage == 'first':
            self.logger.info('Aggregating the day lineage by first...')
            day_first = sorted(list(set(mutation_day_lineage_df['day'])))[0]
            lineage_first = sorted(list(set(mutation_day_lineage_df['lineage'])))[0]
            mutation_day_lineage_df = mutation_day_lineage_df.sort_values(by=['day', 'lineage'], ascending=True)
            mutation_day_lineage_df = mutation_day_lineage_df.groupby('mutation').agg({'day': 'first', 'lineage': 'first'}).reset_index()
            mutation_df = mutation_df.merge(mutation_day_lineage_df, on='mutation', how='left')
            mutation_df['day'] = mutation_df['day'].fillna(day_first)
            mutation_df['lineage'] = mutation_df['lineage'].fillna(lineage_first)

        elif inference_aggregate_day_lineage == 'all':
            self.logger.info('Aggregating the day lineage by all...')
            day_mode = mutation_day_lineage_df['day'].mode()[0]
            lineage_mode = mutation_day_lineage_df['lineage'].mode()[0]
            mutation_df = mutation_df.merge(mutation_day_lineage_df, on='mutation', how='left')
            mutation_df['day'] = mutation_df['day'].fillna(day_mode)
            mutation_df['lineage'] = mutation_df['lineage'].fillna(lineage_mode)

        elif inference_aggregate_day_lineage == None:
            self.logger.info('Not aggregating the day lineage...')

        mutation_df['daystr'] = mutation_df['day'].map(self.day2daystr_dict)
        self.logger.info('The shape of the dataframe of inference data is: {}'.format(mutation_df.shape))

        self.logger.info('Getting dataset for inference...')
        from data_loader.utility_dataset import LoRaDMSInferenceDataset
        test_dataset = self._create_dataset(LoRaDMSInferenceDataset, mutation_df)

        return test_dataset