import torch
import torch.nn as nn
import torch.nn.functional as F


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


class Embedding_based(nn.Module):

    def __init__(self, args, n_users, n_items, n_entities, n_relations):

        super(Embedding_based, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_users = n_users
        self.n_items = n_items
        self.n_entities = n_entities
        self.n_relations = n_relations

        self.KG_embedding_type = args.KG_embedding_type

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.cf_l2loss_lambda = args.cf_l2loss_lambda
        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.user_embed = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embed = nn.Embedding(self.n_items, self.embed_dim)
        nn.init.xavier_uniform_(self.user_embed.weight)
        nn.init.xavier_uniform_(self.item_embed.weight)

        self.entity_embed = nn.Embedding(self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        nn.init.xavier_uniform_(self.entity_embed.weight)
        nn.init.xavier_uniform_(self.relation_embed.weight)

        self.fusion_type = args.fusion_type  # 新增参数，控制融合类型

        # TransR 
        self.trans_M = nn.Parameter(torch.Tensor(self.n_relations, self.embed_dim, self.relation_dim))
        nn.init.xavier_uniform_(self.trans_M)


    def calc_kg_loss_TransR(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                                                # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]                                                           # (kg_batch_size, embed_dim, relation_dim)

        h_embed = self.entity_embed(h)                                                  # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_embed(pos_t)                                          # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_embed(neg_t)                                          # (kg_batch_size, embed_dim)

        # 1. 计算头实体，尾实体和负采样的尾实体在对应关系空间中的投影嵌入
        r_mul_h = torch.bmm(h_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(pos_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(neg_t_embed.unsqueeze(1), W_r).squeeze(1)  # (kg_batch_size, relation_dim)

        # 2. 对关系嵌入，头实体嵌入，尾实体嵌入，负采样的尾实体嵌入进行L2范数归一化
        r_embed = F.normalize(r_embed, p=2, dim=-1)
        r_mul_h = F.normalize(r_mul_h, p=2, dim=-1)
        r_mul_pos_t = F.normalize(r_mul_pos_t, p=2, dim=-1)
        r_mul_neg_t = F.normalize(r_mul_neg_t, p=2, dim=-1)

        # 3. 分别计算正样本三元组 (h_embed, r_embed, pos_t_embed) 和负样本三元组 (h_embed, r_embed, neg_t_embed) 的得分
        pos_score = torch.norm(r_mul_h + r_embed - r_mul_pos_t, p=2, dim=-1)  # (kg_batch_size)
        neg_score = torch.norm(r_mul_h + r_embed - r_mul_neg_t, p=2, dim=-1)  # (kg_batch_size)

        # 4. 使用 BPR Loss 进行优化，尽可能使负样本的得分大于正样本的得分
        kg_loss = torch.mean(F.relu(pos_score - neg_score + 1.0))  # Margin can be adjusted

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(r_mul_neg_t)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss


    def calc_kg_loss_TransE(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)                                                # (kg_batch_size, relation_dim)
        
        h_embed = self.entity_embed(h)                                                  # (kg_batch_size, embed_dim)
        pos_t_embed = self.entity_embed(pos_t)                                          # (kg_batch_size, embed_dim)
        neg_t_embed = self.entity_embed(neg_t)                                          # (kg_batch_size, embed_dim)

        # 5. 对关系嵌入，头实体嵌入，尾实体嵌入，负采样的尾实体嵌入进行L2范数归一化
        r_embed = F.normalize(r_embed, p=2, dim=-1)
        h_embed = F.normalize(h_embed, p=2, dim=-1)
        pos_t_embed = F.normalize(pos_t_embed, p=2, dim=-1)
        neg_t_embed = F.normalize(neg_t_embed, p=2, dim=-1)

        # 6. 分别计算正样本三元组 (h_embed, r_embed, pos_t_embed) 和负样本三元组 (h_embed, r_embed, neg_t_embed) 的得分
        pos_score = torch.norm(h_embed + r_embed - pos_t_embed, p=2, dim=-1)  # (kg_batch_size)
        neg_score = torch.norm(h_embed + r_embed - neg_t_embed, p=2, dim=-1)  # (kg_batch_size)

        # 7. 使用 BPR Loss 进行优化，尽可能使负样本的得分大于正样本的得分
        kg_loss = torch.mean(F.relu(pos_score - neg_score + 1.0))  # Margin can be adjusted

        l2_loss = _L2_loss_mean(h_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(pos_t_embed) + _L2_loss_mean(neg_t_embed)
        loss = kg_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def _fuse_entity_knowledge(self, item_embed, item_kg_embed):
        if self.fusion_type == 'add':  # 相加
            return item_embed + item_kg_embed
        elif self.fusion_type == 'mul':  # 逐元素乘积
            return item_embed * item_kg_embed
        elif self.fusion_type == 'concat':  # 拼接
            concat_embed = torch.cat([item_embed, item_kg_embed], dim=-1)
            # 如果是拼接，则可能需要额外的线性层来调整维度
            # 这里假设 embed_dim 和 relation_dim 是相同的；如果不是，你可能需要调整尺寸
            # linear_layer = nn.Linear(concat_embed.size(-1), self.embed_dim).to(item_embed.device)
            # return linear_layer(concat_embed)
            return concat_embed
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

    def calc_cf_loss(self, user_ids, item_pos_ids, item_neg_ids):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)
        """
        user_embed = self.user_embed(user_ids)                                          # (cf_batch_size, embed_dim)
        item_pos_embed = self.item_embed(item_pos_ids)                                  # (cf_batch_size, embed_dim)
        item_neg_embed = self.item_embed(item_neg_ids)                                  # (cf_batch_size, embed_dim)

        item_pos_kg_embed = self.entity_embed(item_pos_ids)                             # (cf_batch_size, embed_dim)
        item_neg_kg_embed = self.entity_embed(item_neg_ids)                             # (cf_batch_size, embed_dim)
        
        # 8. 为 物品嵌入 注入 实体嵌入的语义信息
        item_pos_cf_embed = self._fuse_entity_knowledge(item_pos_embed, item_pos_kg_embed)
        item_neg_cf_embed = self._fuse_entity_knowledge(item_neg_embed, item_neg_kg_embed)
        
        if self.fusion_type == 'concat':
            user_embed = torch.cat([user_embed,user_embed],dim=1)
        pos_score = torch.sum(user_embed * item_pos_cf_embed, dim=1)                    # (cf_batch_size)
        neg_score = torch.sum(user_embed * item_neg_cf_embed, dim=1)                    # (cf_batch_size)

        cf_loss = (-1.0) * torch.log(1e-10 + F.sigmoid(pos_score - neg_score))
        cf_loss = torch.mean(cf_loss)

        l2_loss = _L2_loss_mean(user_embed) + _L2_loss_mean(item_pos_cf_embed) + _L2_loss_mean(item_neg_cf_embed)
        loss = cf_loss + self.cf_l2loss_lambda * l2_loss
        return loss


    def calc_loss(self, user_ids, item_pos_ids, item_neg_ids, h, r, pos_t, neg_t, only_cf=False, only_kg=False):
        """
        user_ids:       (cf_batch_size)
        item_pos_ids:   (cf_batch_size)
        item_neg_ids:   (cf_batch_size)

        h:              (kg_batch_size)
        r:              (kg_batch_size)
        pos_t:          (kg_batch_size)
        neg_t:          (kg_batch_size)
        """
        if self.KG_embedding_type == 'TransR':
            calc_kg_loss = self.calc_kg_loss_TransR
        elif self.KG_embedding_type == 'TransE':
            calc_kg_loss = self.calc_kg_loss_TransE
        
        if(only_cf):
            cf_loss = self.calc_cf_loss(user_ids, item_pos_ids, item_neg_ids)
            kg_loss = 0
        elif(only_kg):
            kg_loss = calc_kg_loss(h, r, pos_t, neg_t)
            cf_loss = 0
        else:
            cf_loss = self.calc_cf_loss(user_ids, item_pos_ids, item_neg_ids)
            kg_loss = calc_kg_loss(h, r, pos_t, neg_t)

        loss = kg_loss + cf_loss
        return loss


    def calc_score(self, user_ids, item_ids):
        """
        user_ids:  (n_users)
        item_ids:  (n_items)
        """
        user_embed = self.user_embed(user_ids)                                          # (n_users, embed_dim)

        item_embed = self.item_embed(item_ids)                                          # (n_items, embed_dim)
        item_kg_embed = self.entity_embed(item_ids)                                     # (n_items, embed_dim)

        # 9. 为 物品嵌入 注入 实体嵌入的语义信息
        item_cf_embed = self._fuse_entity_knowledge(item_embed, item_kg_embed)          # (n_items, embed_dim)

        if self.fusion_type == 'concat':
            user_embed = torch.cat([user_embed,user_embed], dim=1)   
        cf_score = torch.matmul(user_embed, item_cf_embed.transpose(0, 1))              # (n_users, n_items)
        
        return cf_score


    def forward(self, *input, is_train, only_cf=False, only_kg=False):
        if is_train:
            return self.calc_loss(*input, only_cf=only_cf, only_kg=only_kg)
        else:
            return self.calc_score(*input)


