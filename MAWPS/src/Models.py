import logging
import os
import sys
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from transformers import BertConfig, BertTokenizer, BertModel
import random
from torch.nn.functional import mse_loss
import math

from torch.nn.functional import log_softmax
logger = logging.getLogger(__name__)

# Transformer Parameters
d_model = 1024  # Embedding Size
d_ff = 1024 # FeedForward dimension
d_k = d_v = 128  # dimension of K(=Q), V
n_layers = 6  # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
def get_attn_pad_mask(seq_q, seq_k):
    '''
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    '''
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # [batch_size, 1, len_k], False is masked
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # [batch_size, len_q, len_k]

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        '''
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        '''
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k) # scores : [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9) # Fills elements of self tensor with value where mask is True.
        
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V) # [batch_size, n_heads, len_q, d_v]
        return context, attn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.pe = self.pe.cuda()

    def forward(self, x):
        '''
        x: [seq_len, batch_size, d_model]
        '''
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1,2)  # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1,2)  # V: [batch_size, n_heads, len_v(=len_k), d_v]

        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size, n_heads, seq_len, seq_len]

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context) # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn

class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )
    def forward(self, inputs):
        '''
        inputs: [batch_size, seq_len, d_model]
        '''
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual) # [batch_size, seq_len, d_model]
class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        '''
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        '''
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs) # [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attn, dec_enc_attn

class Batch_Net_large(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net_large, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.LeakyReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.LeakyReLU(True))
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

    def forward(self, input_ids, attention_mask, token_type_ids, labels, getlogit: bool = False):
        bertout = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                                                    token_type_ids=token_type_ids)
        print(bertout)
        token_embeddings = bertout[0]
        
        token_embeddings = token_embeddings[:, 1:-1, :]
        sen_vectors = []
        new_labels = labels[:, 1:].float()

        for idd, lll in enumerate(labels):
            num_vector = torch.unsqueeze(token_embeddings[idd, lll[0], :], 0)
            sen_vectors.append(num_vector)

        sen_vectors = self.dropout(torch.cat(sen_vectors, 0))
        logits = self.fc(sen_vectors)

        if labels is not None:
            loss = self.loss_func(logits, new_labels)

            return logits, loss.mean()
        return logits, None

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.fc.state_dict(), os.path.join(save_dir, 'fc_weight.bin'))



class Batch_Net_CLS(nn.Module):
    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):
        super(Batch_Net_CLS, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1), nn.LeakyReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2), nn.LeakyReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

        # # 对网络的参数进行Xavier初始化
        nn.init.xavier_uniform_(self.layer1[0].weight)
        nn.init.xavier_uniform_(self.layer2[0].weight)
        nn.init.xavier_uniform_(self.layer3[0].weight)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class MwpBertModel_CLS(torch.nn.Module):
    def __init__(self, bert_path_or_config, num_labels, train_loss='MSE', fc_path=None, multi_fc=False,
                 fc_hidden_size: int = 1024):
        super(MwpBertModel_CLS, self).__init__()
        self.num_labels = num_labels
        self.all_iter_num = 5
        if isinstance(bert_path_or_config, str):
            self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config)
        elif isinstance(bert_path_or_config, BertConfig):
            self.bert = BertModel(bert_path_or_config)

        self.dropout = torch.nn.Dropout(0.1)
        self.d_model = 768
        if multi_fc:
            #! 768*2 -> 2048 -> 1024 -> 28
            self.fc = Batch_Net_CLS(self.bert.config.hidden_size * 2, fc_hidden_size, int(fc_hidden_size / 2),
                                    num_labels)
        else:
            self.fc = torch.nn.Linear(in_features=self.bert.config.hidden_size * 2, out_features=num_labels, bias=True)


        # self.tokenizer = BertTokenizer.from_pretrained(bert_path_or_config)
        self.tokenizer = T5Tokenizer.from_pretrained('t5-large')

        #! 28 -> 768/2 -> 768 -> 768*2
        # self.code_emb = torch.nn.Linear(in_features=self.num_labels, out_features=self.d_model*2, bias=True)
        self.code_emb = Batch_Net_CLS(self.num_labels,int(self.d_model/2),self.d_model,self.d_model*2)
        #! 768*2
        # self.code_check = torch.nn.Linear(in_features=self.d_model*2, out_features=2,bias=True)
        self.code_check = Batch_Net_CLS(self.d_model*2,self.d_model,int(self.d_model/2),2)
        #! Transformer decoder self-attention
        self.dec_self_attn = MultiHeadAttention()
        assert train_loss in ['MSE', 'L1', 'Huber']
        if train_loss == 'MSE':
            self.loss_func = torch.nn.MSELoss(reduction='mean')
        elif train_loss == 'L1':
            self.loss_func = torch.nn.L1Loss()
        else:
            self.loss_func = torch.nn.SmoothL1Loss()
        if fc_path:
            self.load(fc_path)

    def forward(self, input_ids, attention_mask, token_type_ids, num_positions, num_codes_labels):
        
        # bertout = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        t5out = self.t5(input_ids = input_ids, attention_mask=attention_mask)
        # token_embeddings = bertout[0]
        # pooler_output = bertout[1]
        token_embeddings = t5out[0]
        pooler_output = t5out[1]
        #! (batch_size, seq_len, d)
        p_hidden = torch.gather(token_embeddings, 1, num_positions.unsqueeze(-1).expand(-1,-1,self.d_model))
        problem_out = pooler_output.unsqueeze(1).expand(p_hidden.shape)
        #! (batch_size, seq_len, 2*d)
        p_hidden = torch.cat((p_hidden,problem_out),dim=-1)
        batch_size = len(num_positions)
        #! (batch_size, output_len, num_labels)
        labels = num_codes_labels.reshape(batch_size,-1, self.num_labels).float()
        pad_vector = torch.Tensor([-1]*self.num_labels).cuda()
        seq_len = labels.shape[1] #!这里指output_len
        ce = nn.CrossEntropyLoss(reduction='mean',ignore_index=-1)#! 忽略pad -1
        pad_mask = torch.any(labels != pad_vector, dim =-1)#! 需要关注的部分code label
        pad_mask = pad_mask.unsqueeze(-1).expand_as(labels)
        
        #! Transformers decoder self-attention
        dec_self_attn_pad_mask = get_attn_pad_mask(num_positions, num_positions)
        
        if self.training:
            # all_loss = 0
            # #! 无mask
            # num_codes = num_codes_labels.reshape(batch_size,-1,self.num_labels)
            # num_codes = self.code_emb(num_codes.float())
            # decoder_inputs = self.dropout(p_hidden + num_codes)
            # decoder_inputs, _ = self.dec_self_attn(decoder_inputs, decoder_inputs, decoder_inputs, dec_self_attn_pad_mask)
            # outputs = self.fc(decoder_inputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
            # all_loss += self.loss_func(labels[pad_mask],outputs[pad_mask])
            
            # #! 判断当前每一个结果是否正确
            # #! 预测check
            # num_codes = num_codes_labels.reshape(batch_size,-1,self.num_labels)
            # code_emb = self.code_emb(outputs.float())
            # code_emb = self.dropout(code_emb + p_hidden)
            # code_check_pred = self.code_check(code_emb.detach())  #[B,Seq_len,2]
            # #! 构造label
            # code_check_label = torch.zeros((batch_size,seq_len,1)).cuda()
            # code_check_label.masked_fill_(torch.all(torch.round(outputs) == num_codes, dim=-1).unsqueeze(-1), torch.scalar_tensor(1))
            # code_check_label.masked_fill_(torch.all(num_codes == pad_vector, dim=-1).unsqueeze(-1), torch.scalar_tensor(-1))
            
            # code_check_label = code_check_label.reshape(-1).long()
            # all_loss += ce(code_check_pred.reshape(-1,2),code_check_label).float()
            
            # # #! 正采样
            # # num_codes = num_codes_labels.reshape(batch_size,-1,self.num_labels)
            # # code_emb = self.code_emb(num_codes.float())
            # # code_emb = self.dropout(code_emb + p_hidden)
            # # # code_check_pred = self.code_check(code_emb.detach())  #[B,Seq_len,2]
            # # code_check_pred = self.code_check(code_emb.detach())  #[B,Seq_len,2]
            # # code_check_label = torch.ones((batch_size,seq_len,1)).cuda()
            # # code_check_label.masked_fill_(torch.all(num_codes == pad_vector, dim=-1).unsqueeze(-1), torch.scalar_tensor(-1))
            # # code_check_label = code_check_label.reshape(-1).long()
            # # all_loss += ce(code_check_pred.reshape(-1,2),code_check_label).float()

            # #! mask 10% ... 60%
            # iter_num = 6
            # mask_count = seq_len
            # for iter in range(iter_num):
            #     #! 随机mask，构造code embedding
            #     # mask_count  -= 1
            #     # if mask_count == 0:
            #     #     break 
            #     num_codes = num_codes_labels.reshape(batch_size,-1,self.num_labels)
            #     # mask_count = max(1,int(0.3 * seq_len))
            #     mask_count = max(1,int((iter+1)*0.1 * seq_len))
            #     mask_index = torch.LongTensor(random.sample(range(seq_len),mask_count))
            #     num_codes.index_fill(1,mask_index.cuda(),torch.Tensor([0]).squeeze().cuda())
            #     num_codes = self.code_emb(num_codes.float())
            #     decoder_inputs = self.dropout(p_hidden + num_codes)
            #     decoder_inputs, _ = self.dec_self_attn(decoder_inputs, decoder_inputs, decoder_inputs, dec_self_attn_pad_mask)
                
            #     outputs = self.fc(decoder_inputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
                
                
            #     all_loss += self.loss_func(labels[pad_mask],outputs[pad_mask])
                
            #     #! 判断当前每一个结果是否正确
            #     #! 预测check
            #     num_codes = num_codes_labels.reshape(batch_size,-1,self.num_labels)
            #     code_emb = self.code_emb(outputs.float())
            #     code_emb =self.dropout(code_emb + p_hidden)
            #     code_check_pred = self.code_check(code_emb.detach())  #[B,Seq_len,2]
            #     #! 构造label
            #     code_check_label = torch.zeros((batch_size,seq_len,1)).cuda()
                
            #     code_check_label.masked_fill_(torch.all(torch.round(outputs) == num_codes, dim=-1).unsqueeze(-1), torch.scalar_tensor(1))
            #     code_check_label.masked_fill_(torch.all(num_codes == pad_vector, dim=-1).unsqueeze(-1), torch.scalar_tensor(-1))
            #     code_check_label = code_check_label.reshape(-1).long()
            #     all_loss += ce(code_check_pred.reshape(-1,2),code_check_label).float()
            
            # #! 全部mask
            # decoder_inputs = self.dropout(p_hidden)
            # decoder_inputs, _ = self.dec_self_attn(decoder_inputs, decoder_inputs, decoder_inputs, dec_self_attn_pad_mask)
            
            # outputs = self.fc(decoder_inputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
            # all_loss += self.loss_func(labels[pad_mask],outputs[pad_mask])
            # #! 判断当前每一个结果是否正确
            # #! 预测check
            # num_codes = num_codes_labels.reshape(batch_size,-1,self.num_labels)
            # code_emb = self.code_emb(outputs.float())
            # code_emb = self.dropout(code_emb + p_hidden)
            # code_check_pred = self.code_check(code_emb.detach())  #[B,Seq_len,2]
            # #! 构造label
            # code_check_label = torch.zeros((batch_size,seq_len,1)).cuda()
            
            # code_check_label.masked_fill_(torch.all(torch.round(outputs) == num_codes, dim=-1).unsqueeze(-1), torch.scalar_tensor(1))
            # code_check_label.masked_fill_(torch.all(num_codes == pad_vector, dim=-1).unsqueeze(-1), torch.scalar_tensor(-1))
            # code_check_label = code_check_label.reshape(-1).long()
            # all_loss += ce(code_check_pred.reshape(-1,2),code_check_label).float()
            # return all_loss
        
            #! 只训练一个单纯的decoder
            decoder_inputs = self.dropout(p_hidden)
            outputs = self.fc(decoder_inputs)
            all_loss = self.loss_func(labels[pad_mask],outputs[pad_mask])
            outputs_round = torch.round(outputs)
            code_emb = self.code_emb(outputs_round.float())
            code_emb = self.dropout(code_emb)
            code_check_pred = self.code_check(code_emb)

            return all_loss 
        else:
            #! 如果是测试
            outputs_list = []

            # iter_num = 1
            # #! 第一次迭代，全部mask
            # decoder_inputs = self.dropout(p_hidden)
            # decoder_inputs, _ = self.dec_self_attn(decoder_inputs, decoder_inputs, decoder_inputs, dec_self_attn_pad_mask)
            
            # outputs = self.fc(decoder_inputs) # dec_logits: [batch_size, tgt_len, tgt_vocab_size]
            # outputs_list.append(outputs)
            # num_codes = num_codes_labels.reshape(batch_size,-1,self.num_labels)
            # code_emb = self.code_emb(outputs.float())
            # code_emb =self.dropout(code_emb + p_hidden)
            # code_check_pred = self.code_check(code_emb)  #[B,Seq_len,2]
            # code_check_pred = log_softmax(code_check_pred,dim=-1)
            # code_check_p = code_check_pred[:,:,1] #[B,Seq_len]
            # # 将pad位置设置为-1，batchsize=1 没有pad
            # # code_check_p.masked_fill_(torch.all(num_codes == pad_vector, dim=-1), torch.scalar_tensor(-1))
            # #! 选取前 (T-t)/T 的进行mask
            # # num_elements_to_mask = int(code_check_p.shape[-1] * (self.all_iter_num - iter_num)/self.all_iter_num)
            # # num_elements_to_mask = code_check_pred.shape[1] - 1
            # num_elements_to_mask = int(code_check_p.shape[-1] * (self.all_iter_num - iter_num)/self.all_iter_num)
            # _ , top_k_indices = torch.topk(code_check_p,k=num_elements_to_mask,dim=-1,largest=False)
            # mask_matrix = torch.zeros((batch_size, seq_len)).cuda()
            # # is_wrong = (code_check_pred[:,:,0]>code_check_pred[:,:,1]).unsqueeze(-1)
            # mask_matrix.scatter_(dim=-1, index=top_k_indices,src=torch.ones_like(top_k_indices).to(mask_matrix.dtype).cuda())
            # mask_matrix = mask_matrix.unsqueeze(-1).expand_as(outputs)
            # #! 迭代中优化
            # #! 循环 mask-predict
            # for i in range(5):
                
            #     #! 暂时放弃零次迭代
            #     # if torch.equal(mask_matrix,torch.zeros_like(mask_matrix)):
            #     #     break
            #     outputs.masked_fill_(mask_matrix.bool(),torch.scalar_tensor(0))
            #     code_emb = self.code_emb(outputs.float())
            #     decoder_inputs = self.dropout(p_hidden + code_emb)
            #     decoder_inputs, _ = self.dec_self_attn(decoder_inputs, decoder_inputs, decoder_inputs, dec_self_attn_pad_mask)
                
            #     predicts = self.fc(decoder_inputs)
            #     # 将is_wrong位置的predicts替换到outputs中
            #     outputs[mask_matrix.bool()] = predicts[mask_matrix.bool()]
                
            #     num_elements_to_mask=int(code_check_p.shape[-1] * (self.all_iter_num - iter_num)/self.all_iter_num)
            #     if num_elements_to_mask == 0:
            #         break
            #     iter_num+=1

            #     code_emb = self.code_emb(outputs.float())
            #     code_emb =self.dropout(code_emb + p_hidden)
            #     code_check_pred = self.code_check(code_emb)  #[B,Seq_len,2]
            #     code_check_pred = log_softmax(code_check_pred,dim=-1)
            #     code_check_p = code_check_pred[:,:,1] #[B,Seq_len]
            #      #! 选取前 (T-t)/T 的进行mask
            #     # num_elements_to_mask = int(code_check_p.shape[-1] * (self.all_iter_num - iter_num)/self.all_iter_num)
                
            #     _ , top_k_indices = torch.topk(code_check_p,k=num_elements_to_mask,dim=-1,largest=False)
            #     mask_matrix = torch.zeros((batch_size, seq_len)).cuda()
            #     # is_wrong = (code_check_pred[:,:,0]>code_check_pred[:,:,1]).unsqueeze(-1)
            #     mask_matrix.scatter_(dim=-1, index=top_k_indices,src=torch.ones_like(top_k_indices).to(mask_matrix.dtype).cuda())
            #     mask_matrix = mask_matrix.unsqueeze(-1).expand_as(outputs)
    
            # outputs_list.append(outputs)
            # # return outputs,code_check_pred,iter_num
            # return outputs_list

            #! 只训练一个单纯的解码器
            decoder_inputs = self.dropout(p_hidden)
            outputs = self.fc(decoder_inputs)
            outputs_list.append(outputs)
            return outputs_list

    def save(self, save_dir):
        self.bert.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        torch.save(self.fc.state_dict(), os.path.join(save_dir, 'fc_weight.bin'))
        torch.save(self.code_emb.state_dict(), os.path.join(save_dir, 'code_emb.bin'))
        torch.save(self.code_check.state_dict(), os.path.join(save_dir, 'code_check.bin'))
        torch.save(self.dec_self_attn.state_dict(), os.path.join(save_dir, 'dec_self_attn.bin'))
    
    def load(self,fc_path):
        self.bert = BertModel.from_pretrained(fc_path)
        self.fc.load_state_dict(torch.load(fc_path+'/fc_weight.bin'))
        self.code_emb.load_state_dict(torch.load(fc_path+'/code_emb.bin'))
        self.code_check.load_state_dict(torch.load(fc_path+'/code_check.bin'))
        self.dec_self_attn.load_state_dict(torch.load(fc_path+'/dec_self_attn.bin'))

class Embedding(nn.Module):
    def __init__(self, input_size, embedding_size, dropout=0.5):
        super(Embedding, self).__init__()

        self.embedding = nn.Embedding(
            input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)

    def forward(self, input_seqs):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        return embedded
    
# The overall pipeline of UTSC_Solver with problem encoder, iterative inference with self-correction mechanism refines mathematical expressions by alternating between the generator and the discriminator. The input is preprocessed mathematical problem text, and the output is the code representation corresponding to an M-Tree structure, which represents the mathematical expression. Within each iteration, we identify errors in the generated expressions using the discriminator. For correct code, the output is labeled as \textit{right}, and for incorrect code, it is labeled as \textit{wrong}. We then replace these erroneous tokens with \texttt{<mask>} and utilize the generator to regenerate them in a non-autoregressive manner. The iteration stops when it reaches the maximum iteration count or when the discriminator outputs all correct results. The final output is then presented as the result.

class MwpBertModel_CLS_classfier(torch.nn.Module):
    def __init__(self, bert_path_or_config, num_labels, train_loss='MSE', fc_path=None, multi_fc=False,
                 fc_hidden_size: int = 1024):
        super(MwpBertModel_CLS_classfier, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path_or_config)
        self.dropout = torch.nn.Dropout(0.2)
        self.d_model = self.bert.config.hidden_size

        #! 768*2 -> 2048 -> 1024 -> 28
        self.fc = Batch_Net_CLS(self.bert.config.hidden_size * 3, fc_hidden_size, int(fc_hidden_size / 2),
                                    num_labels)


        self.get_goal1 = nn.Linear(self.bert.config.hidden_size * 2, fc_hidden_size)
        self.get_goal2 = nn.Linear(self.bert.config.hidden_size * 2, fc_hidden_size)
        self.get_goal3 = nn.Linear(fc_hidden_size, self.bert.config.hidden_size)
        self.get_goal4 = nn.Linear(fc_hidden_size, self.bert.config.hidden_size)

        # self.attn1 = nn.MultiheadAttention(embed_dim=d_model, num_heads=8)
        # self.attn2 = nn.MultiheadAttention(embed_dim=d_model, num_heads=8)
        # self.tokenizer = BertTokenizer.from_pretrained(bert_path_or_config)

        if fc_path:
            self.load(fc_path)

    def forward(self, input_ids, attention_mask, token_type_ids, num_positions, num_codes_labels):
        
        # bertout = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # token_embeddings = bertout[0]
        # pooler_output = bertout[1]
        t5out = self.t5(input_ids = input_ids, attention_mask=attention_mask)
        token_embeddings = t5out[0]
        pooler_output = t5out[1]
        
        p_hidden = torch.gather(token_embeddings, 1, num_positions.unsqueeze(-1).expand(-1,-1,self.d_model))
        p_hidden = self.dropout(p_hidden)
        problem_out = pooler_output
        # problem_out = token_embeddings.mean(1)
        problem_out = problem_out.unsqueeze(1).expand(p_hidden.shape)
        # problem_out = self.dropout(problem_out)
        
        # context, _ = self.attn1(p_hidden.transpose(0,1), token_embeddings.transpose(0,1), token_embeddings.transpose(0,1),key_padding_mask=attention_mask.float())
        # context = context.transpose(0,1)
        
        # goal1 = torch.tanh(self.get_goal1(torch.cat((p_hidden,problem_out),dim=-1)))
        # goal2 = torch.sigmoid(self.get_goal2(torch.cat((p_hidden,problem_out),dim=-1)))
        # goal = goal1 * goal2
        # goal3 = torch.tanh(self.get_goal3(goal))
        # goal4 = torch.sigmoid(self.get_goal4(goal))
        # goal = goal3 * goal4

        # context, _ = self.attn2(goal.transpose(0,1), token_embeddings.transpose(0,1), token_embeddings.transpose(0,1),key_padding_mask=attention_mask.float())
        # context = context.transpose(0,1)

        #! (batch_size, seq_len, 2*d)
        # decoder_inputs = torch.cat((p_hidden,goal),dim=-1)
        # decoder_inputs = torch.cat((p_hidden,context,goal),dim=-1)
        decoder_inputs = torch.cat((p_hidden,problem_out),dim=-1)
        # decoder_inputs = goal
        batch_size = len(num_positions)
        #! (batch_size, output_len, num_labels)
        labels = num_codes_labels.reshape(batch_size,-1, self.num_labels).clone()
        
        #! 必需为0或1
        # labels[labels > 1] = 1 
        labels = labels.float()
        pad_vector = torch.Tensor([-1]*self.num_labels).cuda()
        pad_mask_pre = torch.any(labels != pad_vector, dim =-1)#! 需要关注的部分code label
        pad_mask = pad_mask_pre.unsqueeze(-1).expand_as(labels)
        
        
        #! 只训练一个单纯的decoder
        # decoder_inputs = self.dropout(p_hidden)
        outputs = self.fc(decoder_inputs)
        # binary_outputs = (torch.sigmoid(outputs) > 0.5).float()
        binary_outputs = torch.round(torch.sigmoid(outputs))

        bce_loss = nn.BCEWithLogitsLoss()(outputs[pad_mask], labels[pad_mask])            
        # 总损失为 BCE 损失、数量损失和位置损失的加权和
        # total_loss = bce_loss + 10*count_loss + 5*position_loss
        total_loss = bce_loss 
        outputs = torch.sigmoid(outputs)
        return total_loss,outputs.detach(),decoder_inputs.detach()


    def save(self, save_dir):
        # self.bert.save_pretrained(save_dir)
        # self.tokenizer.save_pretrained(save_dir)
        pass
        # torch.save(self.fc.state_dict(), os.path.join(save_dir, 'fc_weight.bin'))
        # torch.save(self.get_goal1.state_dict(), os.path.join(save_dir, 'get_goal1.bin'))
        # torch.save(self.get_goal2.state_dict(), os.path.join(save_dir, 'get_goal2.bin'))
        # torch.save(self.get_goal3.state_dict(), os.path.join(save_dir, 'get_goal3.bin'))
        # torch.save(self.get_goal4.state_dict(), os.path.join(save_dir, 'get_goal4.bin'))
        # torch.save(self.attn1.state_dict(), os.path.join(save_dir, 'attn1.bin'))
        # torch.save(self.attn2.state_dict(), os.path.join(save_dir, 'attn2.bin'))

    
    def load(self,fc_path):
        # self.bert = BertModel.from_pretrained(fc_path)
        # self.fc.load_state_dict(torch.load(fc_path+'/fc_weight.bin'))
        # self.get_goal1.load_state_dict(torch.load(fc_path+'/get_goal1.bin'))
        # self.get_goal2.load_state_dict(torch.load(fc_path+'/get_goal2.bin'))
        # self.get_goal3.load_state_dict(torch.load(fc_path+'/get_goal3.bin'))
        # self.get_goal4.load_state_dict(torch.load(fc_path+'/get_goal4.bin'))
        # self.attn1.load_state_dict(torch.load(fc_path+'/attn1.bin'))
        # self.attn2.load_state_dict(torch.load(fc_path+'/attn2.bin'))
        pass

class Refiner(nn.Module):
    def __init__(self, num_labels,hidden_size,fc_path=None):
        super(Refiner, self).__init__()
        self.num_labels = num_labels
        # self.checker = Batch_Net_CLS(self.num_labels, check_hidden_size, int(check_hidden_size / 2), 1)
        self.checker =  nn.Sequential(
                        # nn.Linear(768+self.num_labels*10, 256),
                        nn.Linear(d_model*2+self.num_labels, 256),
                        nn.LeakyReLU(),
                        nn.Linear(256, 256),
                        nn.LeakyReLU(),
                        # nn.Linear(256, 128),
                        # nn.LeakyReLU(),
                        # nn.Linear(128, 64),
                        # nn.LeakyReLU(),
                        nn.Linear(256, 1)
                    )
        # self.corrector = Batch_Net_CLS(d_model*2 + self.num_labels, hidden_size, int(hidden_size / 2), self.num_labels)
        # self.ln_code = nn.Linear(self.num_labels, self.num_labels*10)
        # self.ln_phidden = nn.Linear(d_model*2, d_model)

        #! 初始化
        nn.init.xavier_uniform_(self.checker[0].weight)
        nn.init.xavier_uniform_(self.checker[2].weight)
        nn.init.xavier_uniform_(self.checker[4].weight)

        # self.loss_func = nn.BCEWithLogitsLoss(weight=torch.tensor([7/3]))
        self.loss_func = nn.BCEWithLogitsLoss(reduction='sum')
        self.dropout = torch.nn.Dropout(0.1)
        if fc_path:
            self.load(fc_path)
    def forward(self, outputs, num_codes_labels,p_hidden):
        labels = num_codes_labels.reshape(outputs.shape[0],-1, self.num_labels).clone()
        pad_vector = torch.Tensor([-1]*self.num_labels).cuda()
        pad_mask = torch.any(labels != pad_vector, dim =-1)#! 需要关注的部分code label

        #! 检测错误 28 -> 1
        # outpus_processed = 1 - 2 * torch.abs(0.5 - outputs)
        # ln_outputs = self.ln_code(outputs)
        # ln_p_hidden = self.ln_phidden(p_hidden)
        # logits = self.checker(torch.cat((ln_outputs,ln_p_hidden),dim=-1))
        logits = self.checker(torch.cat((outputs,p_hidden),dim=-1))
        outputs_rounded = torch.round(outputs)
        check_labels = torch.eq(outputs_rounded, labels)
        check_labels = torch.all(check_labels, dim=-1, keepdim=True)
        check_labels = check_labels.to(torch.float)
        #! 截断防止NAN
        # logits = torch.clamp(logits, min=1e-7, max=1-1e-7)
        #! 保证正负个数相等
        num_negatives = (check_labels[pad_mask] == 0).sum()
        num_postives = (check_labels[pad_mask] == 1).sum()
        if num_postives > num_negatives:
            positives_indices = (check_labels == 1).nonzero()
            random_indices = torch.randperm(len(positives_indices))
            selected_indices = random_indices[:num_postives-num_negatives]
            selected_positives = positives_indices[selected_indices]
            new_pad_mask = torch.zeros_like(pad_mask)
            new_pad_mask[pad_mask] = 1
            new_pad_mask[selected_positives[:,0], selected_positives[:,1]] = 0

            all_loss = self.loss_func(logits[new_pad_mask],check_labels[new_pad_mask])
        elif num_postives < num_negatives:
            negtives_indices = (check_labels == 0).nonzero()
            random_indices = torch.randperm(len(negtives_indices))
            selected_indices = random_indices[:num_negatives-num_postives]
            selected_negtives = negtives_indices[selected_indices]
            new_pad_mask = torch.zeros_like(pad_mask)
            new_pad_mask[pad_mask] = 1
            new_pad_mask[selected_negtives[:,0], selected_negtives[:,1]] = 0
            all_loss = self.loss_func(logits[new_pad_mask],check_labels[new_pad_mask])
        else:
            all_loss = self.loss_func(logits[pad_mask],check_labels[pad_mask])


        # #! 纠正错误 28 -> 628*2+28 -> hidden -> 28
        # p_hidden = torch.cat((outputs, p_hidden),dim=-1)
        # logits1 = self.corrector(self.dropout(p_hidden))
        # indices = torch.logical_and((1-check_labels).squeeze(-1), pad_mask)
        # labels = labels.to(torch.float)
        # #! 计算loss
        # all_loss = self.loss_func(logits1[pad_mask],labels[pad_mask])
        # all_loss += 10*self.loss_func(logits1[indices],labels[indices])
        # logits1 = torch.sigmoid(logits1)
        # logits1[check_labels.bool().squeeze(-1)] = outputs[check_labels.bool().squeeze(-1)]       
        

        return all_loss,torch.sigmoid(logits),check_labels

    def save(self, save_dir):
        pass
    def load(self,fc_path):
        self.checker.load_state_dict(torch.load(fc_path+'/checker.bin'))
        self.ln_code.load_state_dict(torch.load(fc_path+'/ln_code.bin'))
        self.ln_phidden.load_state_dict(torch.load(fc_path+'/ln_phidden.bin'))

# The overall pipeline of UTSC_Solver with problem encoder, iterative inference with self-correction mechanism refines mathematical expressions by alternating between the generator and the discriminator. The input is preprocessed mathematical problem text, and the output is the code representation corresponding to an M-Tree structure, which represents the mathematical expression. Within each iteration, we identify errors in the generated expressions using the discriminator. For correct code, the output is labeled as \textit{right}, and for incorrect code, it is labeled as \textit{wrong}. We then replace these erroneous tokens with \texttt{<mask>} and utilize the generator to regenerate them in a non-autoregressive manner. The iteration stops when it reaches the maximum iteration count or when the discriminator outputs all correct results. The final output is then presented as the result.

# geneartor The generator is a decoder based on the transformer architecture\textsuperscript{~\cite{vaswani2017attention}}, utilizing a non-autoregressive framework.
class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads = 8, num_layers = 6, max_sequence_length = 768):
        super(Decoder, self).__init__()
        
        self.max_sequence_length = max_sequence_length
        
        self.d_model = d_model
        # Positional encoding layer
        self.positional_encoding = self.get_positional_encoding(d_model, max_sequence_length)
        
        # Transformer decoder layers
        self.decoder_layers = nn.TransformerDecoderLayer(d_model, num_heads,batch_first=True)
        
        # Transformer decoder
        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layers,
            num_layers=num_layers
        )
        
        # Linear layer for output
        self.fc = nn.Linear(d_model, vocab_size)
        
    def forward(self, memory, tgt,tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # Embedding for decoder input
        
        # Add positional encoding decoder_input batch_sirst
        # tgt = tgt  * math.sqrt(self.d_model)
        # tgt += self.positional_encoding[:, :tgt.size(1), :].to(tgt.device)
        # Transformer decoding with cross-attention to bert_output
        output = self.decoder(tgt, memory, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        
        # Linear layer to get logits
        logits = self.fc(output)
        
        return logits

    def get_positional_encoding(self, d_model, max_sequence_length):  
        # Create positional encoding  
        positional_encoding = torch.zeros(max_sequence_length, d_model)  
        row = torch.arange(0, max_sequence_length).reshape(-1,1)
        col = torch.pow(10000, torch.arange(0, d_model, 2) / d_model)
        positional_encoding[:, 0::2] = torch.sin(row / col)  
        positional_encoding[:, 1::2] = torch.cos(row / col)

        return positional_encoding.unsqueeze(0)

class UTSC_Solver(torch.nn.Module):
    def __init__(self, bert_path, num_labels, iter_num, num_layers,hidden_size: int = 1024):
        super(UTSC_Solver, self).__init__()
        self.iter_num = iter_num
        self.num_labels = num_labels
        self.encoder = BertModel.from_pretrained(pretrained_model_name_or_path=bert_path)
        self.dropout = torch.nn.Dropout(0.5)
        self.d_model = self.encoder.config.hidden_size
        self.hidden_size = hidden_size
        self.code_emb = Embedding(self.num_labels, self.d_model, dropout=self.dropout.p)
        self.fusion_fc = nn.Linear(self.d_model * 3, self.d_model)
        self.relu_activation = nn.ReLU()
        self.geneartor = Decoder(vocab_size=self.num_labels,num_layers=num_layers , d_model=self.d_model) 
        self.discriminator = Decoder(vocab_size=2, num_layers=num_layers,d_model=self.d_model) # 二分类器 加上<pad>


        self.get_goal1 = nn.Linear(self.d_model * 2, self.hidden_size)
        self.get_goal2 = nn.Linear(self.d_model * 2, self.hidden_size)
        self.get_goal3 = nn.Linear(self.hidden_size, self.d_model)
        self.get_goal4 = nn.Linear(self.hidden_size, self.d_model)


    def forward(self, input_ids, input_mask, token_type_ids, num_positions, tgt_ids, tgt_mask, problem_id):
        bertout = self.encoder(input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids)
        problem_embeddings = bertout[0]
        pooler_output = bertout[1]
        # 抽取num_positions对应的token
        p_hidden = torch.gather(problem_embeddings, 1, num_positions.unsqueeze(-1).expand(-1,-1,self.d_model))
        problem_out = pooler_output.unsqueeze(1).expand(p_hidden.shape)
        
        # 添加dropout
        p_hidden = self.dropout(p_hidden)
        problem_out = self.dropout(problem_out)

        goal1 = torch.tanh(self.get_goal1(torch.cat((p_hidden,problem_out),dim=-1)))
        goal2 = torch.sigmoid(self.get_goal2(torch.cat((p_hidden,problem_out),dim=-1)))
        goal = goal1 * goal2
        goal3 = torch.tanh(self.get_goal3(goal))
        goal4 = torch.sigmoid(self.get_goal4(goal))
        goal = goal3 * goal4

        all_loss_g, all_loss_d = 0,0
        code_g_input = torch.ones((p_hidden.shape[0], p_hidden.shape[1])).cuda()
        code_g_input = code_g_input.long()
        # (batch_size, seq_len)
        code_pred_list = []
        judgement_pred_list = []
        discriminator_label_list = []
        if self.training:
            for iter in range(self.iter_num):
                #! generator
                codes_embedding = self.code_emb(code_g_input) # 一开始code全为<mask> 
                decoder_inputs = self.fusion_fc(torch.cat((codes_embedding, p_hidden,goal),dim=-1))
                decoder_inputs = self.relu_activation(decoder_inputs)  # ReLU Activation
                decoder_inputs = self.dropout(decoder_inputs)  # Dropout
                code_pred = self.geneartor(tgt=decoder_inputs, memory=problem_embeddings, tgt_key_padding_mask=~(tgt_mask.bool()), memory_key_padding_mask=~(input_mask.bool()))

                index_mask = torch.eq(code_g_input, 1)
                index_mask = index_mask.bool()
                loss_g = nn.functional.cross_entropy(code_pred.transpose(1,2), tgt_ids,ignore_index=0, reduction='none')
                loss_g = loss_g * index_mask
                loss_g = loss_g.sum() / max(1,index_mask.sum())
                all_loss_g += loss_g
                code_pred = torch.argmax(code_pred, dim=-1)

                #! 更新mask位置的code，并更新code_pred_list
                if len(code_pred_list) == 0:
                    code_pred_list.append(code_pred)
                else:
                    code_pred_last = code_pred_list[-1]
                    code_pred[code_g_input != 1] = code_pred_last[code_g_input != 1]
                    code_pred_list.append(code_pred)

                #! discriminator
                discriminator_label = torch.eq(code_pred, tgt_ids).long()
                discriminator_label[~(tgt_mask.bool())] = -100
                discriminator_label_list.append(discriminator_label)
                decoder_inputs = self.fusion_fc(torch.cat((self.code_emb(code_pred), p_hidden,goal),dim=-1))
                decoder_inputs = self.relu_activation(decoder_inputs)  # ReLU Activation
                decoder_inputs = self.dropout(decoder_inputs)  # Dropout
                judgement_pred = self.discriminator(tgt=decoder_inputs, memory=problem_embeddings, tgt_key_padding_mask=~(tgt_mask.bool()), memory_key_padding_mask=~(input_mask.bool()))
                loss_d = nn.functional.cross_entropy(judgement_pred.transpose(1,2), discriminator_label, ignore_index=-100, reduction='mean')
                all_loss_d += loss_d
                judgement_pred = torch.argmax(judgement_pred, dim=-1)
                judgement_pred_list.append(judgement_pred)
                #! 更新code_g_input 对应judgement_pred为1的位置更新为code_pred，错误位置更新为1，使用真实的标签去判断，而不是使用judgement_pred
                new_code_g_input = code_pred.clone()
                new_code_g_input[discriminator_label == 0] = 1
                # new_code_g_input[discriminator_label == 0] = 1
                code_g_input = new_code_g_input

            return all_loss_g, all_loss_d, code_pred_list, judgement_pred_list, discriminator_label_list
        
        else:

            for iter in range(self.iter_num):
                #! generator
                codes_embedding = self.code_emb(code_g_input) # 一开始code全为<mask>
                decoder_inputs = self.fusion_fc(torch.cat((codes_embedding, p_hidden,goal),dim=-1))
                decoder_inputs = self.relu_activation(decoder_inputs)  # ReLU Activation
                decoder_inputs = self.dropout(decoder_inputs)  # Dropout
                code_pred = self.geneartor(tgt=decoder_inputs, memory=problem_embeddings, tgt_key_padding_mask=~(tgt_mask.bool()), memory_key_padding_mask=~(input_mask.bool()))
                # 将输出转化为概率值
                code_pred = torch.softmax(code_pred, dim=-1)
                #! 更新mask位置的code，并更新code_pred_list
                # 将上一轮中code_g_input==1的位置对应的code的code_pred置为0

                if len(code_pred_list) == 0:
                    code_pred = torch.argmax(code_pred, dim=-1)
                    code_pred_list.append(code_pred)
                else:
                    # code_pred_last = code_pred_list[-1]
                    # code_pred = torch.argmax(code_pred, dim=-1)
                    # code_pred[code_g_input != 1] = code_pred_last[code_g_input != 1]
                    
                    # code_pred_list.append(code_pred)

                    # #! 获得上n轮错误的code的三维下标
                    wrong_indices =  torch.nonzero(code_g_input == 1).tolist()
                    wrong_indices = [(i,j) for i,j in wrong_indices]
          
                    for i,j in wrong_indices:
                        for code_pred_last in code_pred_list:
                            code_pred[(i,j,code_pred_last[(i,j)].item())] = 0
                    code_pred = torch.argmax(code_pred, dim=-1)
                    code_pred[code_g_input != 1] = code_pred_last[code_g_input != 1]
                    code_pred_list.append(code_pred)

                    
                    

                #! discriminator
                discriminator_label = torch.eq(code_pred, tgt_ids).long()
                discriminator_label[~(tgt_mask.bool())] = -100
                discriminator_label_list.append(discriminator_label)
                decoder_inputs = self.fusion_fc(torch.cat((self.code_emb(code_pred), p_hidden,goal),dim=-1))
                decoder_inputs = self.relu_activation(decoder_inputs)  # ReLU Activation
                decoder_inputs = self.dropout(decoder_inputs)  # Dropout
                judgement_pred = self.discriminator(tgt=decoder_inputs, memory=problem_embeddings, tgt_key_padding_mask=~(tgt_mask.bool()), memory_key_padding_mask=~(input_mask.bool()))
                judgement_pred = torch.argmax(judgement_pred, dim=-1)
                judgement_pred_list.append(judgement_pred)

                #! 更新code_g_input 对应judgement_pred为0的位置更新为1
                new_code_g_input = code_pred.clone()
                new_code_g_input[discriminator_label == 0] = 1
                # 如果judgement_pred全是1，则不更新
                if torch.all(discriminator_label == 1):
                    break
                else:
                    code_g_input = new_code_g_input
            # return code_pred 
            return code_pred, code_pred_list, judgement_pred_list, discriminator_label_list, iter+1


    def save(self, save_dir):
        pass


    
    def load(self,fc_path):
        pass
