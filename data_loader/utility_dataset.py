# -*- coding: utf-8 -*-

import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from natsort import natsorted
from joblib import Parallel, delayed
from torch.utils.data import Dataset as TorchDataset
    

class TLAttBERTDataset(TorchDataset):
    def __init__(self, logger, 
                       seq2daystr_lineage_dict, 
                       daystr2idx_dict,
                       unique_daystr_list,
                       unique_lineage_list,
                       tokenizer, 
                       max_len, 
                       sample_ratio=0.5,
                       sample_method='day-lineage2seq'):
        self.logger = logger
        self.seq_list = list(seq2daystr_lineage_dict.keys())
        self.seq_list.sort()
        self.seq2daystr_lineage_dict = seq2daystr_lineage_dict
        self.daystr2idx_dict = daystr2idx_dict
        self.unique_daystr_list = unique_daystr_list
        self.unique_lineage_list = unique_lineage_list

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.sample_ratio = sample_ratio
        self.sample_method = sample_method
        self._has_logged_example = False
        np.random.seed(0)

        if self.sample_method == 'day-lineage2seq':
            self.logger.info('Using day-lineage2seq sampling method...')
            self.seqidx2seq_dict, self.seq2seqidx_dict, self.daystr_lineage2seqidx_dict, \
                self.unique_daystr_lineage_list = self.get_daylineage2seq_dist()
        elif self.sample_method == 'sample_seq2day-lineage':
            self.logger.info('Using sample_seq2day-lineage sampling method...')
            self.seqidx2seq_dict, self.seq2seqidx_dict, self.seqidx2daystr_lineage_distr_dict, \
                self.unique_seqidx_list = self.get_seq2daylineage_dist()
        elif self.sample_method == 'top_seq2day-lineage':
            self.logger.info('Using top_seq2day-lineage sampling method...')
            self.seqidx2seq_dict, self.seq2seqidx_dict, self.seqidx_list, self.daystr_list, \
                self.lineage_list, self.unique_seqidx_list = self.get_top_seq2daylineage()


    def get_daylineage2seq_dist(self):
        self.logger.info('Getting the distribution of day-lineage2seq...')
        seqidx2seq_dict = {'seq-'+str(i): seq for i, seq in enumerate(self.seq_list)}
        seq2seqidx_dict = {seq: 'seq-'+str(i) for i, seq in enumerate(self.seq_list)}
        
        daystr_lineage2seqidx_dict = {}
        for seq, daystr_lineage_list in self.seq2daystr_lineage_dict.items():
            for daystr_lineage in list(set(daystr_lineage_list)):
                if daystr_lineage not in daystr_lineage2seqidx_dict:
                    daystr_lineage2seqidx_dict[daystr_lineage] = [seq2seqidx_dict[seq]]
                else:
                    daystr_lineage2seqidx_dict[daystr_lineage].append(seq2seqidx_dict[seq])
        unique_daystr_lineage_list = natsorted(list(daystr_lineage2seqidx_dict.keys()))
        self.logger.info('Number of unique day-lineage: {}'.format(len(daystr_lineage2seqidx_dict)))

        return seqidx2seq_dict, seq2seqidx_dict, daystr_lineage2seqidx_dict, unique_daystr_lineage_list

    def get_seq2daylineage_dist(self):
        self.logger.info('Getting the distribution of seq2day-lineage...')
        self.logger.info('Getting the first collection date of each lineage for ordering...')
        lineage2date_df = pd.read_csv('/home/jiannanyang/project/viral-mutation-pytorch/data/COVID-19/output/Lineage_date.csv')
        lineage2date_dict = dict(zip(lineage2date_df['Lineage_formated'], lineage2date_df['Collection date']))
        lineage2datelineage_dict = {lineage: lineage2date_dict[lineage] + '-' + lineage for lineage in lineage2date_dict}
        datelineage2lineage_dict = {lineage2datelineage_dict[lineage]: lineage for lineage in lineage2date_dict}
        
        seqidx2seq_dict = {'seq-'+str(i): seq for i, seq in enumerate(self.seq2daystr_lineage_dict.keys())}
        seq2seqidx_dict = {seq: 'seq-'+str(i) for i, seq in enumerate(self.seq2daystr_lineage_dict.keys())}
        unique_seqidx_list = list(seqidx2seq_dict.keys())
        unique_seqidx_list.sort()
        
        seqidx2daystr_lineage_distr_dict = {}
        for seq, daystr_lineage_list in tqdm(self.seq2daystr_lineage_dict.items()):
            unique_daystr_lineage_combin_list = list(set(daystr_lineage_list))
            unique_daystr_lineage_combin_list = natsorted([tuple([daystr, lineage2datelineage_dict[lineage]]) 
                                                           for daystr, lineage in unique_daystr_lineage_combin_list])
            unique_daystr_lineage_combin_list = [tuple([daystr, datelineage2lineage_dict[lineage]]) 
                                                 for daystr, lineage in unique_daystr_lineage_combin_list]
            
            idx_list = list(range(len(unique_daystr_lineage_combin_list)))
            all_count = len(daystr_lineage_list)
            daystr_lineage_combin_freq_list = [daystr_lineage_list.count(daystr_lineage) / all_count 
                                               for daystr_lineage in unique_daystr_lineage_combin_list]
            seqidx2daystr_lineage_distr_dict[seq2seqidx_dict[seq]] = (idx_list, unique_daystr_lineage_combin_list, daystr_lineage_combin_freq_list)
        
        return seqidx2seq_dict, seq2seqidx_dict, seqidx2daystr_lineage_distr_dict, unique_seqidx_list 

    def get_top_seq2daylineage(self):
        self.logger.info('Getting the distribution of seq2day-lineage...')
        self.logger.info('Getting the first collection date of each lineage for ordering...')
        lineage2date_df = pd.read_csv('/home/jiannanyang/project/viral-mutation-pytorch/data/COVID-19/output/Lineage_date.csv')
        lineage2date_dict = dict(zip(lineage2date_df['Lineage_formated'], lineage2date_df['Collection date']))
        lineage2datelineage_dict = {lineage: lineage2date_dict[lineage] + '-' + lineage for lineage in lineage2date_dict}
        datelineage2lineage_dict = {lineage2datelineage_dict[lineage]: lineage for lineage in lineage2date_dict}
        
        seqidx2seq_dict = {'seq-'+str(i): seq for i, seq in enumerate(self.seq2daystr_lineage_dict.keys())}
        seq2seqidx_dict = {seq: 'seq-'+str(i) for i, seq in enumerate(self.seq2daystr_lineage_dict.keys())}
        unique_seqidx_list = list(seqidx2seq_dict.keys())
        unique_seqidx_list.sort()
        
        seqidx_list, daystr_list, lineage_list = [], [], []
        for seq, daystr_lineage_list in tqdm(self.seq2daystr_lineage_dict.items()):
            unique_daystr_lineage_combin_list = list(set(daystr_lineage_list))
            unique_daystr_lineage_combin_list = natsorted([tuple([daystr, lineage2datelineage_dict[lineage]]) 
                                                           for daystr, lineage in unique_daystr_lineage_combin_list])
            unique_daystr_lineage_combin_list = natsorted([tuple([daystr, datelineage2lineage_dict[lineage]]) 
                                                           for daystr, lineage in unique_daystr_lineage_combin_list])
            
            idx_list = list(range(len(unique_daystr_lineage_combin_list)))
            all_count = len(daystr_lineage_list)
            daystr_lineage_combin_freq_list = [daystr_lineage_list.count(daystr_lineage) / all_count for daystr_lineage in unique_daystr_lineage_combin_list]
            
            # get the index of top-1 day-lineage
            top_idx = np.argmax(daystr_lineage_combin_freq_list)
            daystr, lineage = unique_daystr_lineage_combin_list[top_idx]
            seqidx_list.append(seq2seqidx_dict[seq])
            daystr_list.append(daystr)
            lineage_list.append(lineage)
            
            # get the index of top-10 day-lineage
            if len(unique_daystr_lineage_combin_list) < self.sample_ratio:
                top_idx_list = np.random.choice(idx_list, self.sample_ratio, p=daystr_lineage_combin_freq_list)
            else:
                top_idx_list = np.argsort(daystr_lineage_combin_freq_list)[::-1][:10]
            for idx in top_idx_list:
                seqidx_list.append(seq2seqidx_dict[seq])
                daystr, lineage = unique_daystr_lineage_combin_list[idx]
                daystr_list.append(daystr)
                lineage_list.append(lineage)
        
        return seqidx2seq_dict, seq2seqidx_dict, seqidx_list, daystr_list, lineage_list, unique_seqidx_list 

    def __len__(self):
        if self.sample_method == 'day-lineage2seq':
            return len(self.seq2daystr_lineage_dict) + len(self.unique_daystr_lineage_list) * self.sample_ratio
        elif self.sample_method == 'sample_seq2day-lineage':
            return len(self.seq2daystr_lineage_dict) + len(self.seqidx2daystr_lineage_distr_dict) * self.sample_ratio
        elif self.sample_method == 'top_seq2day-lineage':
            return len(self.seqidx_list)
        
    def get_item_daylineage2seq(self, idx):
        # first, go through the list of sequences and days, and use the first one
        if idx < len(self.seq2daystr_lineage_dict):
            if idx == 0:
                self.logger.info('Going through the list of sequences and days...')
            seq = self.seq_list[idx]
            daystr, lineage = self.seq2daystr_lineage_dict[seq][0]
        else:
            if idx == len(self.seq2daystr_lineage_dict):
                self.logger.info('Randomly sampling sequences and days...')
            # sample_ratio is the interval, 0 - sample_ratio -> 0, sample_ratio - 2*sample_ratio -> 1, ...
            daystr_lineage_idx = (idx - len(self.seq2daystr_lineage_dict)) // self.sample_ratio
            daystr, lineage = self.unique_daystr_lineage_list[daystr_lineage_idx]
            seqidx = np.random.choice(self.daystr_lineage2seqidx_dict[tuple([daystr, lineage])], 1)[0]
            seq = self.seqidx2seq_dict[seqidx]

        token_list = [daystr, lineage] + list(seq)
        output = self.tokenizer(" ".join(token_list),
                                truncation=True, 
                                max_length=self.max_len,
                                padding='max_length',
                                return_special_tokens_mask=True,
                                return_tensors="pt")
        input_ids = output['input_ids']
        attention_mask = output['attention_mask']
        token_type_ids = output['token_type_ids']

        day_list = [self.daystr2idx_dict[daystr]]
        day_tensor = torch.tensor(day_list, dtype=torch.long)

        if not self._has_logged_example:
            self.logger.info("Example of tokenized input: {} -> {}".format(
                daystr + " " + lineage + " " + seq, input_ids))
            self._has_logged_example = True

        return {"input_ids": input_ids.squeeze(),
                "attention_mask": attention_mask.squeeze(),
                "token_type_ids": token_type_ids.squeeze(),
                "time_ids": day_tensor}
    
    def get_item_sampleseq2daylineage(self, idx):
        # first, go through the list of sequences and days, and use the first one
        if idx < len(self.unique_seqidx_list):
            if idx == 0:
                self.logger.info('Going through the list of sequences and days...')
            seqidx = self.unique_seqidx_list[idx]
            daystr, lineage = self.seqidx2daystr_lineage_distr_dict[seqidx][1][0]
            seq = self.seqidx2seq_dict[seqidx]
        else:
            if idx == len(self.unique_seqidx_list):
                self.logger.info('Randomly sampling sequences and days...')
            # sample_ratio is the interval, 0 - sample_ratio -> 0, sample_ratio - 2*sample_ratio -> 1, ...
            seqidx = self.unique_seqidx_list[(idx - len(self.seqidx2daystr_lineage_distr_dict)) // self.sample_ratio]
            idx_list, unique_daystr_lineage_combin_list, daystr_lineage_combin_freq_list \
                = self.seqidx2daystr_lineage_distr_dict[seqidx]
            daystr_lineage_combin_idx = np.random.choice(idx_list, 1, p=daystr_lineage_combin_freq_list)[0]
            daystr, lineage = unique_daystr_lineage_combin_list[daystr_lineage_combin_idx]
            seq = self.seqidx2seq_dict[seqidx]

        token_list = [daystr, lineage] + list(seq)
        output = self.tokenizer(" ".join(token_list),
                                truncation=True, 
                                max_length=self.max_len,
                                padding='max_length',
                                return_special_tokens_mask=True,
                                return_tensors="pt")
        input_ids = output['input_ids']
        attention_mask = output['attention_mask']
        token_type_ids = output['token_type_ids']

        day_list = [self.daystr2idx_dict[daystr]]
        day_tensor = torch.tensor(day_list, dtype=torch.long)

        if not self._has_logged_example:
            self.logger.info("Example of tokenized input: {} -> {}".format(
                daystr + " " + lineage + " " + seq, input_ids))
            self._has_logged_example = True

        return {"input_ids": input_ids.squeeze(),
                "attention_mask": attention_mask.squeeze(),
                "token_type_ids": token_type_ids.squeeze(),
                "time_ids": day_tensor}
    
    def get_item_topseq2daylineage(self, idx):
        seqidx = self.seqidx_list[idx]
        daystr = self.daystr_list[idx]
        lineage = self.lineage_list[idx]
        seq = self.seqidx2seq_dict[seqidx]

        token_list = [daystr, lineage] + list(seq)
        output = self.tokenizer(" ".join(token_list),
                                truncation=True, 
                                max_length=self.max_len,
                                padding='max_length',
                                return_special_tokens_mask=True,
                                return_tensors="pt")
        input_ids = output['input_ids']
        attention_mask = output['attention_mask']
        token_type_ids = output['token_type_ids']

        day_list = [self.daystr2idx_dict[daystr]]
        day_tensor = torch.tensor(day_list, dtype=torch.long)

        if not self._has_logged_example:
            self.logger.info("Example of tokenized input: {} -> {}".format(
                daystr + " " + lineage + " " + seq, input_ids))
            self._has_logged_example = True

        return {"input_ids": input_ids.squeeze(),
                "attention_mask": attention_mask.squeeze(),
                "token_type_ids": token_type_ids.squeeze(),
                "time_ids": day_tensor}

    def __getitem__(self, idx):
        if self.sample_method == 'day-lineage2seq':
            return self.get_item_daylineage2seq(idx)
        elif self.sample_method == 'sample_seq2day-lineage':
            return self.get_item_sampleseq2daylineage(idx)
        elif self.sample_method == 'top_seq2day-lineage':
            return self.get_item_topseq2daylineage(idx)


class LoRaDMSDataset(TorchDataset):
    def __init__(self, logger, 
                       dms_df, 
                       ref_seq, 
                       tokenizer, 
                       max_len,
                       prepend_time=True,
                       prepend_lineage=True,
                       time_attention=True,
                       daystr2idx_dict=None,
                       n_targets=10,
                       n_samples=10):
        self.logger = logger
        self.dms_df = dms_df
        self.ref_seq = ref_seq
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prepend_time = prepend_time
        self.prepend_lineage = prepend_lineage
        self.time_attention = time_attention
        self.daystr2idx_dict = daystr2idx_dict
        self._has_logged_example = False

        self.input_ids_final, self.attention_masks_final, self.outcomes_final, \
            self.idxs_final, self.time_ids_final = self._preprocess_dms(dms_df, n_targets, n_samples)

    def _prepare_seq_for_tokenization(self, unique_seq_df):
        if self.prepend_time and self.prepend_lineage:
            unique_seq_df['train_seq'] = unique_seq_df['daystr'] + " " + unique_seq_df['lineage'] + " " + unique_seq_df['seq'].apply(lambda x: " ".join(list(x)))
            seq_list = unique_seq_df['train_seq'].tolist()
        elif self.prepend_time:
            unique_seq_df['train_seq'] = unique_seq_df['daystr'] + " " + unique_seq_df['seq'].apply(lambda x: " ".join(list(x)))
            seq_list = unique_seq_df['train_seq'].tolist()
        elif self.prepend_lineage:
            unique_seq_df['train_seq'] = unique_seq_df['lineage'] + " " + unique_seq_df['seq'].apply(lambda x: " ".join(list(x)))
            seq_list = unique_seq_df['train_seq'].tolist()
        else:
            seq_list = unique_seq_df['seq'].apply(lambda x: " ".join(list(x))).tolist()
        return seq_list
    
    def _parallel_tokenizer_chunk(self, seq_chunk):
        return self.tokenizer(
            seq_chunk,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_special_tokens_mask=True,
            return_tensors="pt")
    
    def _sample_and_replace(self, outcome_l, idx_l, n):
        index_l = [i for i, value in enumerate(outcome_l) if not np.isnan(value)]
        result_outcome_l = []
        result_idx_l = []
        while index_l:
            sampled_index_l = random.sample(index_l, min(n, len(index_l)))

            sampled_outcome_l = outcome_l.copy()
            for i in range(len(sampled_outcome_l)):
                if i not in sampled_index_l:
                    sampled_outcome_l[i] = np.nan
            result_outcome_l.append(sampled_outcome_l)

            sampled_idx_l = idx_l.copy()
            for i in range(len(sampled_idx_l)):
                if i not in sampled_index_l:
                    sampled_idx_l[i] = np.nan
            result_idx_l.append(sampled_idx_l)
            index_l = [i for i in index_l if i not in sampled_index_l]

        return result_outcome_l, result_idx_l

    def _preprocess_dms(self, dms_df, n_targets, n_samples):
        self.logger.info("Preprocessing DMS data...")
        dms_df = dms_df.rename(columns={'target': 'outcome'})
        # codes from https://github.com/TheSatoLab/CoVFit.
        nans = [np.nan] * n_targets    
        unique_seq_ids = sorted(dms_df['seq_id'].unique())
        outcome_sum_d = {seq_id: nans.copy() for seq_id in unique_seq_ids}
        idx_sum_d = {seq_id: nans.copy() for seq_id in unique_seq_ids}

        for index, row in dms_df.iterrows():
            seq_id, task_id, outcome = row['seq_id'], row['task_id'], row['outcome']
            outcome_sum_d[seq_id][task_id] = outcome
            idx_sum_d[seq_id][task_id] = index

        unique_seq_df = dms_df.drop_duplicates(subset='seq_id')[['seq_id','seq', 'daystr', 'lineage']]
        unique_seq_df = unique_seq_df.sort_values(by='seq_id')
        seq_list = self._prepare_seq_for_tokenization(unique_seq_df)
        self.logger.info('Example of seq_list: {}'.format(seq_list[0][:10]))

        # Split sequences into chunks for parallel processing
        self.logger.info("Tokenizing sequences...")
        n_chunks = 10
        seq_chunks = [list(chunk) for chunk in np.array_split(seq_list, n_chunks)]
        encodings = Parallel(n_jobs=-1)(delayed(self._parallel_tokenizer_chunk)(seq_chunk) for seq_chunk in seq_chunks)
        self.logger.info("Tokenization completed.")
        # Combine the results from all chunks
        input_ids = torch.cat([encoding["input_ids"] for encoding in encodings], dim=0)
        attention_masks = torch.cat([encoding["attention_mask"] for encoding in encodings], dim=0)
        if not self._has_logged_example:
            self.logger.info("Example of tokenized input: {} -> {}".format(
                seq_list[0][:20], input_ids[0][:10]))
            self._has_logged_example = True

        outcomes = [outcome_sum_d[seq_id] for seq_id in unique_seq_ids]
        idxs = [idx_sum_d[seq_id] for seq_id in unique_seq_ids]
        time_ids = []
        for seq_id in unique_seq_ids:
            time_ids.append(self.daystr2idx_dict[unique_seq_df[unique_seq_df['seq_id'] == seq_id]['daystr'].values[0]])

        input_ids_final = []
        attention_masks_final = []
        outcomes_final = []
        idxs_final = []
        time_ids_final = []

        for i in range(len(input_ids)):
            input_id = input_ids[i]
            attention_mask = attention_masks[i]
            outcome = outcomes[i]
            idx = idxs[i]
            outcome2, idx2 = self._sample_and_replace(outcome, idx, n_samples)
            input_id2 = [input_id] * len(outcome2)
            attention_mask2 = [attention_mask] * len(outcome2)
            time_id = [time_ids[i]] * len(outcome2)
            input_ids_final.extend(input_id2)
            attention_masks_final.extend(attention_mask2)
            outcomes_final.extend(outcome2)
            idxs_final.extend(idx2)
            time_ids_final.extend(time_id)

        outcomes_final = torch.tensor(outcomes_final, dtype=torch.float32)
        idxs_final = torch.tensor(idxs_final, dtype=torch.float32)
        time_ids_final = torch.tensor(time_ids_final, dtype=torch.long)
        return input_ids_final, attention_masks_final, outcomes_final, idxs_final, time_ids_final

    def __len__(self):
        return len(self.outcomes_final)

    def __getitem__(self, idx):
        if self.time_attention:
            item = {
                'input_ids': self.input_ids_final[idx],
                'attention_mask': self.attention_masks_final[idx],
                'labels': self.outcomes_final[idx],
                'weights': None,
                'time_ids': self.time_ids_final[idx]
            }
        else:
            item = {
                'input_ids': self.input_ids_final[idx],
                'attention_mask': self.attention_masks_final[idx],
                'labels': self.outcomes_final[idx],
                'weights': None
            }
        return item


class LoRaDMSInferenceDataset(TorchDataset):
    def __init__(self, logger, 
                       dms_df, 
                       tokenizer, 
                       max_len,
                       prepend_time=True,
                       prepend_lineage=True,
                       time_attention=True,
                       daystr2idx_dict=None,
                       ref_seq=None,
                       n_targets=None,
                       n_samples=None):
        self.logger = logger
        self.mutation_df = dms_df
        self.mutation_df = self.mutation_df.sort_values(by='mutation')
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.prepend_time = prepend_time
        self.prepend_lineage = prepend_lineage
        self.time_attention = time_attention
        self.daystr2idx_dict = daystr2idx_dict
        self._has_logged_example = False

        self.input_ids_final, self.attention_masks_final, self.outcomes_final, \
            self.time_ids_final = self._preprocess_mutation_df(self.mutation_df)

    def _prepare_seq_for_tokenization(self, unique_seq_df):
        if self.prepend_time and self.prepend_lineage:
            unique_seq_df['train_seq'] = unique_seq_df['daystr'] + " " + unique_seq_df['lineage'] + " " + unique_seq_df['seq'].apply(lambda x: " ".join(list(x)))
            seq_list = unique_seq_df['train_seq'].tolist()
        elif self.prepend_time:
            unique_seq_df['train_seq'] = unique_seq_df['daystr'] + " " + unique_seq_df['seq'].apply(lambda x: " ".join(list(x)))
            seq_list = unique_seq_df['train_seq'].tolist()
        elif self.prepend_lineage:
            unique_seq_df['train_seq'] = unique_seq_df['lineage'] + " " + unique_seq_df['seq'].apply(lambda x: " ".join(list(x)))
            seq_list = unique_seq_df['train_seq'].tolist()
        else:
            seq_list = unique_seq_df['seq'].apply(lambda x: " ".join(list(x))).tolist()
        return seq_list
    
    def _parallel_tokenizer_chunk(self, seq_chunk):
        return self.tokenizer(
            seq_chunk,
            truncation=True,
            max_length=self.max_len,
            padding='max_length',
            return_special_tokens_mask=True,
            return_tensors="pt")

    def _preprocess_mutation_df(self, mutation_df):
        self.logger.info("Preprocessing mutation data...")
        seq_list = self._prepare_seq_for_tokenization(mutation_df)

        # Split sequences into chunks for parallel processing
        self.logger.info("Tokenizing sequences...")
        n_chunks = 10
        seq_chunks = [list(chunk) for chunk in np.array_split(seq_list, n_chunks)]
        encodings = Parallel(n_jobs=-1)(delayed(self._parallel_tokenizer_chunk)(seq_chunk) for seq_chunk in seq_chunks)
        self.logger.info("Tokenization completed.")
        # Combine the results from all chunks
        input_ids = torch.cat([encoding["input_ids"] for encoding in encodings], dim=0)
        attention_masks = torch.cat([encoding["attention_mask"] for encoding in encodings], dim=0)
        if not self._has_logged_example:
            self.logger.info("Example of tokenized input: {} -> {}".format(
                seq_list[0][:20], input_ids[0][:10]))
            self._has_logged_example = True
        
        time_ids, outcomes = [], []
        for i, row in mutation_df.iterrows():
            time_ids.append(self.daystr2idx_dict[row['daystr']])
            outcomes.append(row['count'])

        time_ids_final = torch.tensor(time_ids, dtype=torch.long)
        outcomes_final = torch.tensor(outcomes, dtype=torch.float32)
        return input_ids, attention_masks, outcomes_final, time_ids_final

    def __len__(self):
        return len(self.outcomes_final)

    def __getitem__(self, idx):
        if self.time_attention:
            item = {
                'input_ids': self.input_ids_final[idx],
                'attention_mask': self.attention_masks_final[idx],
                'labels': None,
                'weights': None,
                'time_ids': self.time_ids_final[idx]
            }
        else:
            item = {
                'input_ids': self.input_ids_final[idx],
                'attention_mask': self.attention_masks_final[idx],
                'labels': None,
                'weights': None
            }
        return item