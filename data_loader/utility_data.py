import pickle
import pandas as pd
from Bio import SeqIO
from pathlib import Path
from dateutil.parser import parse as dparse
from data_loader.utility import species2group, country2continent


def load_cov_data(logger, data_dir, cut_off_start=None, cut_off_end=None, get_ref_only=False):
    def split_seqs(seqs):
        train_seqs, test_seqs = {}, {}

        logger.info('Splitting seqs...')
        for idx, seq in enumerate(seqs):
            if idx % 100 < 2:
                test_seqs[seq] = seqs[seq]
            else:
                train_seqs[seq] = seqs[seq]
        logger.info('{} train seqs, {} test seqs.'
            .format(len(train_seqs), len(test_seqs)))

        return train_seqs, test_seqs

    def parse_gisaid(entry):
        fields = entry.split('|')
        
        strain = fields[1]
        collection_date = pd.to_datetime(fields[2], errors='ignore')
        if collection_date is None or type(collection_date) != pd._libs.tslibs.timestamps.Timestamp:
            return None
        
        if (cut_off_start != None and cut_off_end != None) and \
            (collection_date < cut_off_start or collection_date >= cut_off_end):
            return None
        
        host = fields[6]
        if host not in species2group:
            group = 'unknown'
        else:
            group = species2group[host].lower()
        try:
            country = fields[10]
            continent = country2continent[country]
        except:
            country = 'unknown'
            continent = 'unknown'
            
        meta = {
            'strain': strain,
            'collection_date': collection_date,
            'host': host,
            'group': group,
            'country': country,
            'continent': continent
        }
        return meta

    def fasta_load(data_dir, seqid2lineage_dict, seqs_all,):
        fasta_data = SeqIO.parse(data_dir, 'fasta')
        for record in fasta_data:
            if len(record.seq) < 1000 or len(record.seq) > 1500:
                continue
            if str(record.seq).count('X') > 0:
                continue
            
            meta = parse_gisaid(record.description)
            # only keep human sequences
            if meta == None or meta['group'] != 'human':
                continue

            aa_sequence = str(record.seq).replace('*', '')
            if aa_sequence not in seqs_all:
                seqs_all[aa_sequence] = []
            
            accession_id = record.description.split('|')[3]
            if accession_id in seqid2lineage_dict:
                meta['lineage'] = seqid2lineage_dict[accession_id]
            else:
                continue
            meta['accession_id'] = accession_id

            seqs_all[aa_sequence].append(meta)

        return seqs_all
    
    def get_ref_seq():
        logger.info('Getting reference sequence...')
        # the reference sequence is collected from SeqIO.read('data/cov/cov2_spike_wt.fasta', 'fasta').seq
        seq = SeqIO.read('/home/jiannanyang/project/viral-mutation-pytorch/data/COVID-19/output/cov2_spike_wt.fasta', 'fasta').seq
        return str(seq)
    
    def get_lineage(year, lineage2formated_dict):
        meta_dir = Path('/home/jiannanyang/project/RSV/viral_mutation_Science/COVID-19/GISAID-spiders/data_process/output/meta_new')
        meta_df = pd.read_csv(meta_dir.joinpath('meta_{}.csv'.format(year)))
        meta_df['Lineage'] = meta_df['Lineage'].map(lineage2formated_dict)
        meta_df = meta_df.dropna(subset=['Lineage'])
        # meta_df['Lineage'] = meta_df['Lineage'].str.replace('.', '')
        seqid2lineage_dict = dict(zip(meta_df['Accession ID'], meta_df['Lineage']))
        seqid2lineage_dict = {k: v.replace('.', '') for k, v in seqid2lineage_dict.items()}
        return seqid2lineage_dict
    
    if get_ref_only:
        logger.info('Getting reference sequence only...')
        ref_seq = get_ref_seq()
        return None, None, ref_seq
    
    logger.info('Loading covid data...')    
    ref_seq = get_ref_seq()

    data_dir = data_dir.joinpath('covid_{}_{}.pkl'.format('2020-01-01', '2023-12-31'))
    if data_dir.exists():
        logger.info('Loading saved data...')
        with open(data_dir, 'rb') as f:
            seqs_all = pickle.load(f)
        logger.info('{} sequences loaded.'.format(len(seqs_all)))
    else:
        logger.info('Loading all data...')
        lineage2formated_df = pd.read_csv('/home/jiannanyang/project/viral-mutation-pytorch/data/COVID-19/output/Lineage_formated.csv')
        lineage2formated_dict = dict(zip(lineage2formated_df['Lineage'], lineage2formated_df['Lineage_formated']))
        seqs_all = {}
        year_list = [2020, 2021, 2022, 2023]
        count_num = 0
        raw_data_dir = Path("/home/jiannanyang/project/RSV/viral_mutation_Science/COVID-19/GISAID_download/output")
        for year in year_list:
            seqid2lineage_dict = get_lineage(year, lineage2formated_dict)
            seqs_all = fasta_load(raw_data_dir.joinpath('spikeprot0117_clean_{}.fasta'.format(year)), 
                                  seqid2lineage_dict,
                                  seqs_all)
            logger.info('{} sequences in {}'.format(len(seqs_all) - count_num, year))
            count_num = len(seqs_all)
        logger.info('In total, {} sequences loaded.'.format(len(seqs_all)))

        with open(data_dir, 'wb') as f:
            pickle.dump(seqs_all, f)

    if cut_off_start != None and cut_off_end != None:
        logger.info('Using cut-off time: {} to {}'.format(cut_off_start, cut_off_end))
        if cut_off_start == dparse('2020-01-01') and cut_off_end == dparse('2023-12-31'):
            return seqs_all, split_seqs, ref_seq
        else:
            seqs_filter = {}
            for seq in seqs_all:
                metas = seqs_all[seq]
                metas_filter = [meta for meta in metas if meta['collection_date'] >= cut_off_start and meta['collection_date'] < cut_off_end]
                if len(metas_filter) > 0:
                    seqs_filter[seq] = metas_filter
            logger.info('{} sequences loaded.'.format(len(seqs_filter)))
            return seqs_filter, split_seqs, ref_seq
    else:
        logger.info('Using cut-off time: 2020-01-01 to 2023-12-31.')
        return seqs_all, split_seqs, ref_seq