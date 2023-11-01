import logging
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import BertConfig, BertTokenizer, BertModel

from torch.nn.functional import mse_loss

logger = logging.getLogger(__name__)


class Batch_Net_large(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net_large, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.LeakyReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.LeakyReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MwpBertModel(torch.nn.Module):
    def __init__(self, bert_path_or_config, num_labels, train_loss='MSE', fc_path=None, multi_fc=False,
                 fc_hidden_size: int = 1024):
        super(MwpBertModel, self).__init__()
        self.num_labels = num_labels

        if isinstance(bert_path_or_config, str):
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config)
        elif isinstance(bert_path_or_config, BertConfig):
            self.bert = BertModel(bert_path_or_config)
        self.dropout = torch.nn.Dropout(0.1)

        if multi_fc:
            self.fc = Batch_Net_large(self.bert.config.hidden_size, fc_hidden_size, int(fc_hidden_size / 2), num_labels)
        else:
            self.fc = torch.nn.Linear(in_features=self.bert.config.hidden_size, out_features=num_labels, bias=True)
        if fc_path:
            self.fc.load_state_dict(torch.load(fc_path))

        self.tokenizer = BertTokenizer.from_pretrained(bert_path_or_config)

        assert train_loss in ['MSE', 'L1', 'Huber']
        if train_loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif train_loss == 'L1':

            self.loss_func = torch.nn.L1Loss()
        else:
            self.loss_func = torch.nn.SmoothL1Loss()
        
        #! 设置不可训练
        self.fc.requires_grad_(False)
        self.bert.requires_grad_(False)

        self.code_emb = torch.nn.Linear(in_features=num_labels, out_features=100, bias=True)
        self.num_emb = torch.nn.Linear(in_features=2*self.bert.config.hidden_size, out_features=400, bias=True)
        self.discriminator = Batch_Net_large(100+400, 1024, int(2014 / 2), 2)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, getlogit: bool = False):
        token_embeddings, pooler_output = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids)

        token_embeddings = token_embeddings[:, 1:-1, :]
        sen_vectors = []
        new_labels = labels[:, 1:].float()

        for idd, lll in enumerate(labels):
            num_vector = torch.unsqueeze(token_embeddings[idd, lll[0], :], 0)
            sen_vectors.append(num_vector)

        sen_vectors = self.dropout(torch.cat(sen_vectors, 0))
        logits = self.fc(sen_vectors)

        if self.training:
            loss = self.loss_func(logits, new_labels)
            return logits, loss.mean()
        

        else:
            return logits, None

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.fc.state_dict(), os.path.join(save_dir, 'fc_weight.bin'))

    def get_sens_vec(self, sens: list):
        self.bert.eval()
        res = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens, pad_to_max_length=True,
                                               return_tensors="pt", max_length=self.max_length)
        input_ids = res["input_ids"]
        attention_mask = res["attention_mask"]
        token_type_ids = res["token_type_ids"]

        logger.info("input ids shape: {},{}".format(input_ids.shape[0], input_ids.shape[1]))
        tensor_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
        sampler = SequentialSampler(tensor_dataset)
        data_loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=self.batch_size)

        all_sen_vec = []
        with torch.no_grad():
            for idx, batch_data in enumerate(data_loader):
                logger.info("get sentences vector: {}/{}".format(idx + 1, len(data_loader)))
                batch_data = [i.to(self.device) for i in batch_data]
                token_embeddings, pooler_output = self.bert(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                            token_type_ids=batch_data[2])
                sen_vecs = []
                for pooling_mode in self.pooling_modes:
                    if pooling_mode == "cls":
                        sen_vec = pooler_output
                    elif pooling_mode == "mean":

                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = input_mask_expanded.sum(1)
                        sum_mask = torch.clamp(sum_mask, min=1e-9)
                        sen_vec = sum_embeddings / sum_mask
                    elif pooling_mode == "max":
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        token_embeddings[input_mask_expanded == 0] = -1e9
                        sen_vec = torch.max(token_embeddings, 1)[0]
                    sen_vecs.append(sen_vec)
                sen_vec = torch.cat(sen_vecs, 1)

                all_sen_vec.append(sen_vec.to("cpu").numpy())
        self.bert.train()
        return np.vstack(all_sen_vec)


class Batch_Net_CLS(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net_CLS, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.BatchNorm1d(n_hidden_1), nn.LeakyReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.BatchNorm1d(n_hidden_2), nn.LeakyReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MwpBertModel_CLS(torch.nn.Module):
    def __init__(self, bert_path_or_config, num_labels, train_loss='MSE', fc_path=None, multi_fc=False,
                 fc_hidden_size: int = 1024,disc_path = None,corrector_path=None):
        super(MwpBertModel_CLS, self).__init__()
        self.num_labels = num_labels

        self.dropout = torch.nn.Dropout(0.1)
        if isinstance(bert_path_or_config, str):
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config)
        if multi_fc:
            self.fc = Batch_Net_CLS(self.bert.config.hidden_size * 2, fc_hidden_size, int(fc_hidden_size / 2), num_labels)
        else:
            self.fc = torch.nn.Linear(in_features=self.bert.config.hidden_size * 2, out_features=num_labels, bias=True)
        

        self.tokenizer = BertTokenizer.from_pretrained(bert_path_or_config)

        assert train_loss in ['MSE', 'L1', 'Huber']
        if train_loss == 'MSE':
            self.loss_func = torch.nn.MSELoss()
        elif train_loss == 'L1':

            self.loss_func = torch.nn.L1Loss()
        else:
            self.loss_func = torch.nn.SmoothL1Loss()
        
        self.code_emb = torch.nn.Linear(in_features=num_labels, out_features=100, bias=True)
        self.num_emb = torch.nn.Linear(in_features=2*self.bert.config.hidden_size, out_features=400, bias=True)
        self.discriminator = self.model = nn.Sequential(
            nn.Linear(100+400, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid())
        self.bceloss = nn.BCELoss()

        

        #! 修改器
        self.code_emb1 = torch.nn.Linear(in_features=num_labels, out_features=128, bias=True)
        self.corrector = Batch_Net_CLS(128 + 768*2, fc_hidden_size, int(fc_hidden_size / 2), num_labels)

        #! 加载模型
        if fc_path:
            self.fc.load_state_dict(torch.load(fc_path))
        
        if disc_path is not None:
            self.code_emb.load_state_dict(torch.load(disc_path+'/code_emb_weight.bin'))
            self.num_emb.load_state_dict(torch.load(disc_path+'/num_emb_weight.bin'))
            self.discriminator.load_state_dict(torch.load(disc_path+'/discriminator_weight.bin'))
        
        if isinstance(bert_path_or_config, str):
            self.bert1 = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config) 
        if corrector_path is not None:
            self.code_emb1.load_state_dict(torch.load(corrector_path+'/code_emb1_weight.bin'))
            self.corrector.load_state_dict(torch.load(corrector_path+'/corrector_weight.bin'))
            self.bert1 = BertModel.from_pretrained(pretrained_model_name_or_path=corrector_path)
        
        #! 设置不可训练
        self.fc.requires_grad_(False)
        self.bert.requires_grad_(False)

        # self.code_emb.requires_grad_(False)
        # self.num_emb.requires_grad_(False)
        # self.discriminator.requires_grad_(False)

        self.code_emb1.requires_grad_(False)
        self.corrector.requires_grad_(False)
        self.bert1.requires_grad_(False)

    def forward(self, input_ids, attention_mask, token_type_ids, labels, getlogit: bool = False):
        #! 避免BN层在前向推理时发生变化
        self.fc.eval()


        bertout = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        token_embeddings, pooler_output = bertout[0], bertout[1]
        token_embeddings = token_embeddings[:, 1:-1, :]
        sen_vectors = []
        new_labels = labels[:, 1:].float()
        for idd, lll in enumerate(labels):
            num_vector = torch.unsqueeze(token_embeddings[idd, lll[0], :], 0)
            sen_vectors.append(num_vector)
        sen_vectors = torch.cat(sen_vectors, 0)
        # 通过[CLS]包含了问题的语义信息
        sen_vectors = torch.cat([sen_vectors, pooler_output], 1)
        sen_vectors = self.dropout(sen_vectors)
        logits = self.fc(sen_vectors)

        #! bert1
        # bertout = self.bert1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # token_embeddings, pooler_output = bertout[0], bertout[1]
        # token_embeddings = token_embeddings[:, 1:-1, :]
        # sen_vectors1 = []
        # for idd, lll in enumerate(labels):
        #     num_vector = torch.unsqueeze(token_embeddings[idd, lll[0], :], 0)
        #     sen_vectors1.append(num_vector)
        # sen_vectors1 = torch.cat(sen_vectors1, 0)
        # # 通过[CLS]包含了问题的语义信息
        # sen_vectors1 = torch.cat([sen_vectors1, pooler_output], 1)
        # sen_vectors1 = self.dropout(sen_vectors1)
        
        
        if self.training:
            #! 训练生成器
            # loss_g = self.loss_func(logits, new_labels)
            # loss_g = loss_g.mean()
            
            #! 训练判别器
            # 根据生成器生成的假数据
            code_pred = torch.round(logits)
            # 判断code_pred与new_labels是否完全一致，注意最后一个维度是28，若最后一个维度完全一样，则为1，否则为0
            real_label = torch.eq(code_pred, new_labels).all(dim=-1,keepdim=True).float()
            # 制造假数据，对于real_label==1的数据，需要修改code_pred（28维度的向量）为假的结果
            for i in range(real_label.shape[0]):
                if real_label[i][0] == 1:
                    # 随机打乱顺序
                    code_pred[i] = code_pred[i][torch.randperm(code_pred.shape[-1])]
            fake_code_emb = self.code_emb(code_pred)
            fake_num_emb = self.num_emb(sen_vectors)
            fake_emb = torch.cat([fake_code_emb, fake_num_emb], 1)
            fake_out = self.discriminator(fake_emb)
            fake_label = torch.zeros(fake_out.shape[0], 1).to(fake_out.device)
            fake_loss = self.bceloss(fake_out, fake_label) 
            # 正例
            true_code_emb = self.code_emb(new_labels)
            true_num_emb = self.num_emb(sen_vectors)
            true_emb = torch.cat([true_code_emb, true_num_emb], 1)
            true_out = self.discriminator(true_emb)
            true_label = torch.ones(true_out.shape[0], 1).to(true_out.device)
            true_loss = self.bceloss(true_out, true_label)
            loss_d = true_loss + fake_loss 

            #! 训练修改器，将label打乱顺序
            # code_wrong = new_labels.clone()
            # for i in range(code_wrong.shape[0]):
            #     code_wrong[i] = code_wrong[i][torch.randperm(code_wrong.shape[-1])]
            # code_wrong_emb = self.code_emb1(code_wrong)
            # corrector_emb = torch.cat([code_wrong_emb, sen_vectors1], 1)
            # corrector_out = self.corrector(corrector_emb)
            # loss_c = self.loss_func(corrector_out, new_labels)
            # loss_c = loss_c.mean()

            
            loss_g = torch.tensor(0)
            # loss_d = torch.tensor(0)
            loss_c = torch.tensor(0)
            return logits, loss_g , loss_d, loss_c
        

        else:
            #构造生成器的真实输出 与label做比较作为鉴别器的测试集

            code_pred = torch.round(logits)
            real_label = torch.eq(code_pred, new_labels).all(dim=-1,keepdim=True).float()


            discriminator_pred = self.discriminator(torch.cat([self.code_emb(code_pred), self.num_emb(sen_vectors)], 1))

            # corrector_pred = self.corrector(torch.cat([self.code_emb1(code_pred), sen_vectors1], 1))
            # corrector_pred = corrector_pred.round()
            corrector_pred = None

            return code_pred, None, discriminator_pred, real_label, corrector_pred
        

    def save(self, save_dir):

        # self.bert.save_pretrained(save_dir)
        # self.tokenizer.save_pretrained(save_dir)
        # torch.save(self.fc.state_dict(), os.path.join(save_dir, 'fc_weight.bin'))

        torch.save(self.discriminator.state_dict(), os.path.join(save_dir, 'discriminator_weight.bin'))
        torch.save(self.code_emb.state_dict(), os.path.join(save_dir, 'code_emb_weight.bin'))
        torch.save(self.num_emb.state_dict(), os.path.join(save_dir, 'num_emb_weight.bin'))

        # torch.save(self.corrector.state_dict(), os.path.join(save_dir, 'corrector_weight.bin'))
        # torch.save(self.code_emb1.state_dict(), os.path.join(save_dir, 'code_emb1_weight.bin'))
        # self.bert1.save_pretrained(save_dir)
        pass

    def get_sens_vec(self, sens: list):
        self.bert.eval()
        res = self.tokenizer.batch_encode_plus(batch_text_or_text_pairs=sens, pad_to_max_length=True,
                                               return_tensors="pt", max_length=self.max_length)
        input_ids = res["input_ids"]
        attention_mask = res["attention_mask"]
        token_type_ids = res["token_type_ids"]

        logger.info("input ids shape: {},{}".format(input_ids.shape[0], input_ids.shape[1]))
        tensor_dataset = TensorDataset(input_ids, attention_mask, token_type_ids)
        sampler = SequentialSampler(tensor_dataset)
        data_loader = DataLoader(tensor_dataset, sampler=sampler, batch_size=self.batch_size)

        all_sen_vec = []
        with torch.no_grad():
            for idx, batch_data in enumerate(data_loader):
                logger.info("get sentences vector: {}/{}".format(idx + 1, len(data_loader)))
                batch_data = [i.to(self.device) for i in batch_data]
                token_embeddings, pooler_output = self.bert(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                            token_type_ids=batch_data[2])
                sen_vecs = []
                for pooling_mode in self.pooling_modes:
                    if pooling_mode == "cls":
                        sen_vec = pooler_output
                    elif pooling_mode == "mean":

                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                        sum_mask = input_mask_expanded.sum(1)
                        sum_mask = torch.clamp(sum_mask, min=1e-9)
                        sen_vec = sum_embeddings / sum_mask
                    elif pooling_mode == "max":
                        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                        token_embeddings[input_mask_expanded == 0] = -1e9
                        sen_vec = torch.max(token_embeddings, 1)[0]
                    sen_vecs.append(sen_vec)
                sen_vec = torch.cat(sen_vecs, 1)

                all_sen_vec.append(sen_vec.to("cpu").numpy())
        self.bert.train()
        return np.vstack(all_sen_vec)
