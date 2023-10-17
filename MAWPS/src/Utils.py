import os
import sys
import json
import torch
from transformers import BertTokenizer
import random
from tqdm import tqdm


class MWPDatasetLoader(object):
    def __init__(self, data, batch_size, shuffle, tokenizer: BertTokenizer, sort=False):
        self.data = data
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.sort = sort
        self.tokenizer = tokenizer

        self.reset()

    def reset(self, doshuffle: bool = False):
        self.examples = self.preprocess(self.data)
        # (sen_tokens, x_len, token_type_id, problem_id, num_codes_labels, num_positions)
        if self.sort:
            self.examples = sorted(self.examples, key=lambda x: x[1], reverse=True)
        if self.shuffle or doshuffle:
            indices = list(range(len(self.examples)))
            random.shuffle(indices)
            self.examples = [self.examples[i] for i in indices]
        self.features = [self.examples[i:i + self.batch_size] for i in range(0, len(self.examples), self.batch_size)]


    def preprocess(self, data):
        """ Preprocess the data and convert to ids. """
        # (problem_id, sentence_list, num_codes_labels, num_positions)
        processed = []
        for d in data:
            problem_id, sentence_list, num_codes_labels, num_positions = d
            sen_tokens = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + sentence_list + ['[SEP]'])
            x_len = len(sen_tokens)
            token_type_id = [0] * x_len
            for pp in num_positions:  #数值的位置token_type_id为1
                token_type_id[pp] = 1
            processed.append((sen_tokens, x_len, token_type_id, problem_id, num_codes_labels, num_positions))
        return processed

    def get_long_tensor(self, tokens_list, batch_size, mask=None):
        """ Convert list of list of tokens to a padded LongTensor. """
        token_len = max(len(x) for x in tokens_list)
       
        tokens = torch.LongTensor(batch_size, token_len).fill_(0) #* 填充<PAD>
        mask_ = torch.LongTensor(batch_size, token_len).fill_(0)
        for i, s in enumerate(tokens_list):
            tokens[i, :len(s)] = torch.LongTensor(s)
            if mask:
                mask_[i, :len(s)] = torch.tensor([1] * len(s), dtype=torch.long)
        if mask:
            return tokens, mask_
        return tokens

    def sort_all(self, batch, lens):
        """ Sort all fields by descending order of lens, and return the original indices. """

        unsorted_all = [lens] + [range(len(lens))] + list(batch)
        sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
        return sorted_all[2:], sorted_all[1]

    def __len__(self):

        return len(self.features)

    def __getitem__(self, index):
        """ Get a batch with index. """
        if not isinstance(index, int):
            raise TypeError
        if index < 0 or index >= len(self.features):
            raise IndexError

        batch = self.features[index]
        batch_size = len(batch)

        # (sen_tokens, x_len, token_type_id, problem_id, num_codes_labels, num_positions)
        batch = list(zip(*batch))# 解压缩序列
        lens = [len(x) for x in batch[0]]
        batch, orig_idx = self.sort_all(batch, lens)
        chars = batch[0]
        input_ids, input_mask = self.get_long_tensor(chars, batch_size, mask=True)
        token_type_ids = self.get_long_tensor(batch[2], batch_size)
        num_positions = self.get_long_tensor(batch[5],batch_size)
        tgt_ids, tgt_mask = self.get_long_tensor(batch[4],batch_size,mask=True)
        problem_id = torch.LongTensor(batch[3])
        return (input_ids, input_mask, token_type_ids, problem_id, num_positions, tgt_ids, tgt_mask)


def process_dataset(file, label2id_data_path: str, max_len: int, lower: bool):
    label2id_or_value = {}
    with open(label2id_data_path, 'r', encoding='utf-8')as ff:
        label2id_list = json.load(ff)
    for inde, label in enumerate(label2id_list):
        label2id_or_value[label] = inde

    processed_dataset = []
    if isinstance(file, list):
        dataset = file
    elif isinstance(file, str):
        with open(file, 'r', encoding='utf-8') as fr:
            dataset = json.load(fr)
    else:
        print('file parameter wrong !!! \n exit !!!')
        sys.exit()

    print('total input dataset len: {}'.format(len(dataset)))

    passed_count = 0
    pbar = tqdm(dataset)
    for idd, raw_mwp in enumerate(pbar):

        pppp_dadd = process_one_mawps_no_None(raw_mwp, label2id_or_value, max_len, lower=lower)
        if pppp_dadd is not None:
            processed_dataset.append(pppp_dadd)
        else:
            passed_count += 1
    print('after process dataset len: {}'.format(len(processed_dataset)))
    print('total passed: {}'.format(passed_count))

    return processed_dataset


def process_one_mawps(raw_mwp: dict, label2id_or_value: dict, max_len: int, lower: bool):
    if lower:
        sentence_list = raw_mwp['T_question_2'].lower().split(' ')
    else:
        sentence_list = raw_mwp['T_question_2'].split(' ')
    if len(sentence_list) > (max_len - 2):
        return None

    num_codes_labels = []
    all_num_postions = []

    num_codes = raw_mwp['num_codes']
    for kk, vv in raw_mwp['T_number_map'].items():
        if kk not in num_codes:

            postions_in_q = []
            for cindex, value in enumerate(sentence_list):
                if kk == value:
                    postions_in_q.append(cindex)

            if len(postions_in_q) == 0:
                print('Can not find num !!! Wrong !!!!!!')
                sys.exit()
            for position in postions_in_q:
                label2id = label2id_or_value["None"]
                label_vector = [0] * len(label2id_or_value)
                label_vector[label2id] = 1
                label_vector = [position] + label_vector

                num_codes_labels.append(label_vector)
            all_num_postions += postions_in_q
        else:

            postions_in_q = []
            for cindex, value in enumerate(sentence_list):
                if kk == value:
                    postions_in_q.append(cindex)

            if len(postions_in_q) == 0:
                print('Can not find num !!! Wrong !!!!!!')
                sys.exit()



            elif len(postions_in_q) == 1:
                position = postions_in_q[0]
                num_marks = num_codes[kk]
                label_vector = [0] * len(label2id_or_value)
                for mark in num_marks:
                    markid = label2id_or_value[mark]
                    label_vector[markid] += 1
                label_vector = [position] + label_vector
                num_codes_labels.append(label_vector)

            else:
                position = postions_in_q[-1]
                num_marks = num_codes[kk]
                label_vector = [0] * len(label2id_or_value)
                for mark in num_marks:
                    markid = label2id_or_value[mark]
                    label_vector[markid] += 1
                label_vector = [position] + label_vector

                num_codes_labels.append(label_vector)

                for pp in postions_in_q[:-1]:
                    label2id = label2id_or_value["None"]
                    label_vector = [0] * len(label2id_or_value)
                    label_vector[label2id] = 1
                    label_vector = [pp] + label_vector
                    num_codes_labels.append(label_vector)

            all_num_postions += postions_in_q

    assert len(num_codes_labels) == len(all_num_postions)
    return (sentence_list, num_codes_labels, all_num_postions)


def process_one_mawps_for_test(raw_mwp: dict, label2id_or_value: dict, max_len: int, lower: bool,
                               tokenizer: BertTokenizer):
    data_mwp = process_one_mawps(raw_mwp, label2id_or_value, max_len, lower)

    if data_mwp is None:
        return None

    processed = []
    sen_tokens = tokenizer.convert_tokens_to_ids(['[CLS]'] + data_mwp[0] + ['[SEP]'])

    x_len = len(sen_tokens)
    pad_len = 0
    if x_len < 100 and False:
        pad_len = (100 - x_len)
        sen_tokens = sen_tokens + [0] * pad_len
    x_len_new = len(sen_tokens)

    token_type_id = [0] * x_len_new
    attention_mask = [1] * x_len + [0] * pad_len
    for pp in data_mwp[2]:
        token_type_id[pp + 1] = 1
    for label in data_mwp[1]:
        processed.append((sen_tokens, attention_mask, token_type_id, label))
    return processed

def process_one_mawps_no_None(raw_mwp: dict, label2id_or_value: dict, max_len: int, lower: bool):
    if lower:
        sentence_list = raw_mwp['T_question_2'].lower().split(' ')
    else:
        sentence_list = raw_mwp['T_question_2'].split(' ')
    if len(sentence_list) > (max_len - 2):
        return None

    num_codes_labels = []
    problem_id = raw_mwp['id']
    num_positions = []

    num_codes = raw_mwp['num_codes']
    for kk,vv in raw_mwp['num_codes'].items():
        for cindex, value in enumerate(sentence_list):
                if kk == value:
                    postions_in_q = cindex
                    break
        if len(vv) == 1:
            num_positions.append(postions_in_q+1)
            num_codes_labels.append(label2id_or_value[vv[0]])
        else:
            vv = [label2id_or_value[v] for v in vv]
            sorted_vv = sorted(vv)
            for v in sorted_vv:
                num_positions.append(postions_in_q+1)
                num_codes_labels.append(v)
    # for kk, vv in raw_mwp['T_number_map'].items():

    #     if kk not in num_codes:
    #         postions_in_q = []
    #         for cindex, value in enumerate(sentence_list):
    #             if kk == value:
    #                 postions_in_q.append(cindex)
    #         for position in postions_in_q:
    #             noneid = label2id_or_value["None"]
    #             num_positions.append(position+1)
    #             label_vector = [0] * len(label2id_or_value)
                
    #             label_vector[noneid] += 1
    #             num_codes_labels = num_codes_labels + label_vector
        
    #     else:
    #         postions_in_q = []
    #         for cindex, value in enumerate(sentence_list):
    #             if kk == value:
    #                 postions_in_q.append(cindex)

    #         if len(postions_in_q) == 0:
    #             print('Can not find num !!! Wrong !!!!!!')
    #             sys.exit()

    #         elif len(postions_in_q) == 1:
    #             position = postions_in_q[0]
    #             num_marks = num_codes[kk]
    #             label_vector = [0] * len(label2id_or_value)
    #             for mark in num_marks:
    #                 markid = label2id_or_value[mark]
    #                 label_vector[markid] += 1
    #             num_codes_labels = num_codes_labels + label_vector
    #             num_positions.append(position+1)

    #         else:
    #             position = postions_in_q[-1]
    #             num_marks = num_codes[kk]
    #             label_vector = [0] * len(label2id_or_value)
    #             for mark in num_marks:
    #                 markid = label2id_or_value[mark]
    #                 label_vector[markid] += 1
                
    #             num_codes_labels = num_codes_labels + label_vector
    #             num_positions.append(position+1)
                
    #             #! 对于其他数字，要求其生成None
    #             for position in postions_in_q[0:-1]:
    #                 label_vector = [0] * len(label2id_or_value)
    #                 noneid = label2id_or_value["None"]
    #                 num_positions.append(position+1)
    #                 label_vector[noneid] += 1
    #                 num_codes_labels = num_codes_labels + label_vector
    
    return (problem_id, sentence_list, num_codes_labels, num_positions)


def process_one_mawps_for_test_no_None(raw_mwp: dict, label2id_or_value: dict, max_len: int, lower: bool,
                               tokenizer: BertTokenizer):
    problem_id, sentence_list, num_codes_labels, num_positions = process_one_mawps_no_None(raw_mwp, label2id_or_value, max_len, lower)

    processed = []
    sen_tokens = tokenizer.convert_tokens_to_ids(['[CLS]'] + sentence_list + ['[SEP]'])

    x_len = len(sen_tokens)
    pad_len = 0
    

    token_type_id = [0] * x_len
    attention_mask = [1] * x_len + [0] * pad_len
    for pp in num_positions:
        token_type_id[pp] = 1
        
    processed.append((sen_tokens, attention_mask, token_type_id, problem_id,num_positions,num_codes_labels))
    return processed