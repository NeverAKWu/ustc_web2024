import os
import random
import collections

import torch
import numpy as np
import pandas as pd

from data_loader.loader_base import DataLoaderBase


class DataLoader(DataLoaderBase):

    def __init__(self, args, logging):
        super().__init__(args, logging)

        self.cf_batch_size = args.cf_batch_size
        self.kg_batch_size = args.kg_batch_size
        self.test_batch_size = args.test_batch_size

        kg_data = self.load_kg(self.kg_file)
        self.construct_data(kg_data)
        self.print_info(logging)


    def construct_data(self, kg_data):
        '''
            kg_data 为 DataFrame 类型
        '''
        # 1. 为KG添加逆向三元组，即对于KG中任意三元组(h, r, t)，添加逆向三元组 (t, r+n_relations, h)，
        #    并将原三元组和逆向三元组拼接为新的DataFrame，保存在 self.kg_data 中。
        # 计算原始关系数量
        original_relations = kg_data['r'].nunique()
        self.n_relations = original_relations

        # 创建逆向三元组并更新关系ID
        reverse_kg_data = kg_data.copy()
        reverse_kg_data[['h', 't']] = reverse_kg_data[['t', 'h']]
        reverse_kg_data['r'] = reverse_kg_data['r'] + self.n_relations  # 为逆向关系分配新ID
        
        # 拼接原三元组和逆向三元组
        self.kg_data = pd.concat([kg_data, reverse_kg_data], ignore_index=True)

        # 2. 计算关系数，实体数和三元组的数量（去重）
        self.n_relations = self.kg_data['r'].nunique()

        entities = set(self.kg_data['h']).union(set(self.kg_data['t']))
        self.n_entities = len(entities)
        self.n_kg_data = len(self.kg_data)
        
        # 3. 根据 self.kg_data 构建字典 self.kg_dict ，其中key为h, value为tuple(t, r)，
        #    和字典 self.relation_dict，其中key为r, value为tuple(h, t)。
        self.kg_dict = collections.defaultdict(list)
        self.relation_dict = collections.defaultdict(list)
        
        # 构建 kg_dict
        for head, group in self.kg_data.groupby('h'):
            for _, row in group.iterrows():
                self.kg_dict[head].append((row['t'], row['r']))
        
        # 构建 relation_dict
        for relation, group in self.kg_data.groupby('r'):
            for _, row in group.iterrows():
                self.relation_dict[relation].append((row['h'], row['t']))
        
        # 确保字典中的列表是唯一的
        for key in self.kg_dict:
            self.kg_dict[key] = list(set(self.kg_dict[key]))
        for key in self.relation_dict:
            self.relation_dict[key] = list(set(self.relation_dict[key]))
        # 确保数据均为int
        self.kg_data = self.kg_data.astype({
            'h': int,
            'r': int,
            't': int
        })
        
# conda activate ml24;python ./main_Embedding_based.py



    def print_info(self, logging):
        logging.info('n_users:      %d' % self.n_users)
        logging.info('n_items:      %d' % self.n_items)
        logging.info('n_entities:   %d' % self.n_entities)
        logging.info('n_relations:  %d' % self.n_relations)

        logging.info('n_cf_train:   %d' % self.n_cf_train)
        logging.info('n_cf_test:    %d' % self.n_cf_test)

        logging.info('n_kg_data:    %d' % self.n_kg_data)


