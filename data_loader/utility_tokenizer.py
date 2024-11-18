# -*- coding: utf-8 -*-

import tempfile
from pathlib import Path
from transformers import BertTokenizer
    

class AATimeLineageTokenizer(object):
    def __init__(self, logger, unique_daystr_list, unique_lineage_list, no_mask_token=None):
        self.PAD = "$"
        self.MASK = "."
        self.UNK = "?"
        self.SEP = "|"
        self.CLS = "*"

        self.logger = logger
        self.unique_daystr_list = unique_daystr_list
        self.unique_lineage_list = unique_lineage_list
        self.no_mask_token = no_mask_token
        self.token_with_special_list, self.token2index_dict = self._get_vocab_dict()

    def _get_vocab_dict(self):
        amino_acids_list = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
                            'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
                            'Y', 'V', 'X', 'Z', 'J', 'U', 'B', 'Z']
        special_tokesn = [self.PAD, self.MASK, self.UNK, self.SEP, self.CLS]

        token_list = amino_acids_list + self.unique_daystr_list + self.unique_lineage_list + special_tokesn
        token2index_dict = {t: i for i, t in enumerate(token_list)}

        return token_list, token2index_dict

    def get_tokenizer(self, max_len=64, tokenizer_dir=None, tokenizer_savedir=None):
        if tokenizer_dir is not None:
            self.logger.info('Loading pre-trained tokenizer...')
            tok = BertTokenizer.from_pretrained(tokenizer_dir)
            return tok

        with tempfile.TemporaryDirectory() as tempdir:
            self.logger.info('Creating tokenizer...')
            vocab_fname = self._write_vocab(self.token2index_dict, Path(tempdir).joinpath("vocab.txt"))
            tok = BertTokenizer(
                vocab_fname,
                do_lower_case=False,
                do_basic_tokenize=True,
                tokenize_chinese_chars=False,
                pad_token=self.PAD,
                mask_token=self.MASK,
                unk_token=self.UNK,
                sep_token=self.SEP,
                cls_token=self.CLS,
                model_max_length=max_len,
                padding_side="right")
            # add self.unique_daystr_list and self.unique_lineage_list as special tokens
            if self.no_mask_token:
                tok.add_special_tokens({"additional_special_tokens": self.no_mask_token})
            if tokenizer_savedir is not None:
                tok.save_pretrained(tokenizer_savedir)
        return tok

    def split(self, seq):
        return list(seq)

    def _write_vocab(self, vocab, fname):
        """
        Write the vocabulary to the fname, one entry per line
        Mostly for compatibility with transformer BertTokenizer
        """
        with open(fname, "w") as sink:
            for v in vocab:
                sink.write(v + "\n")
        return fname