# -*- coding: utf-8 -*-
"""
@Time : 2021/12/30 18:38
@Author : WangBin
@File : Train.py 
@Software: PyCharm 
"""
import json
import logging
import os
import numpy as np
import sys
import torch
import random
import datetime
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

# from src.MwpDataset import MwpDataSet
from src.Utils import process_dataset, MWPDatasetLoader
from src.Models import MwpBertModel, MwpBertModel_CLS
from src.Evaluation import eval_multi_clf

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def beijing(sec, what):
    beijing_time = datetime.datetime.now() + datetime.timedelta(hours=8)
    return beijing_time.timetuple()
def train(args):
    # devices setting
    setup_seed(0)
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # directory
    args.output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    best_model_dir = os.path.join(args.output_dir, "best_model")
    os.makedirs(best_model_dir, exist_ok=True)
    latest_model_dir = os.path.join(args.output_dir, "latest_model")
    os.makedirs(latest_model_dir, exist_ok=True)
    log_dir = os.path.join(args.output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # logging setting to file and screen
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    logging.Formatter.converter = beijing
    handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    # start training
    # with open(args.label2id_path, 'r', encoding='utf-8') as f:
    #     label2id = json.load(f)

    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    # train data...
    # (self, cached_path: str, file_path: str, tokenizer: BertTokenizer,
    # label2id_data_path: str, max_len: int = 510, has_label: bool = True)

    if args.train_type != 'together':
        # train data...
        logger.info("get train data loader...")
        examplesss_train = process_dataset(file=args.train_data_path, label2id_data_path=args.label2id_path,
                                           max_len=args.train_max_len, lower=True)

        if args.train_type == 'one-by-one-in-same-batch':
            # one by one 的话 shuffle 洗牌 和 按长度降序排序 都要为 False
            train_data_loader = MWPDatasetLoader(data=examplesss_train, batch_size=args.batch_size, shuffle=False,
                                                 tokenizer=tokenizer, seed=72, sort=False)
        elif args.train_type == 'one-by-one-random':
            train_data_loader = MWPDatasetLoader(data=examplesss_train, batch_size=args.batch_size, shuffle=True,
                                                 tokenizer=tokenizer, seed=72, sort=False)
        else:
            print('args.train_type wrong!!!')
            sys.exit()

        # dev data...
        logger.info("get dev data loader...")
        examplesss_test = process_dataset(file=args.dev_data_path, label2id_data_path=args.label2id_path,
                                          max_len=args.test_dev_max_len, lower=True)

        dev_data_loader = MWPDatasetLoader(data=examplesss_test, batch_size=args.batch_size, shuffle=False,
                                           tokenizer=tokenizer, seed=72, sort=False)
    else:
        print('args.train_type == together not yet!!!')
        sys.exit()

    total_steps = int(len(train_data_loader) * args.num_epochs)
    steps_per_epoch = len(train_data_loader)

    if args.warmup < 1:
        warmup_steps = int(total_steps * args.warmup)
    else:
        warmup_steps = int(args.warmup)

    # model
    logger.info("define model...")
    if not args.use_cls:
        model = MwpBertModel(bert_path_or_config=args.pretrain_model_path, num_labels=args.num_labels,
                             fc_path=args.fc_path, multi_fc=args.multi_fc, train_loss=args.train_loss,
                             fc_hidden_size=args.fc_hidden_size)
    else:
        model = MwpBertModel_CLS(bert_path_or_config=args.pretrain_model_path, num_labels=args.num_labels,
                                 fc_path=args.fc_path, multi_fc=args.multi_fc, train_loss=args.train_loss,
                                 fc_hidden_size=args.fc_hidden_size,disc_path = args.discriminator_path,corrector_path = args.corrector_path)


    model.to(args.device)
    model.zero_grad()
    model.train()

    # optimizer
    logger.info("define optimizer...")
    no_decay = ["bias", "LayerNorm.weight"]
    paras = dict(model.named_parameters())
    # logger.info("===========================train setting parameters=========================")
    # for n, p in paras.items():
    #     logger.info("{}-{}".format(n, str(p.shape)))
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        }, {
            "params": [p for n, p in paras.items() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    # train
    best_acc = -1
    
    args.log_steps = int(steps_per_epoch / 10)

    log_best_acc = os.path.join(best_model_dir, 'best-acc.txt')
    ff_best = open(log_best_acc, 'a', encoding='utf-8')
    log_test_acc = os.path.join(latest_model_dir, 'test-acc.txt')
    ff_test = open(log_test_acc, 'a', encoding='utf-8')

    #预测
    if args.mode == 'test':
        logger.info("\n>>>>>>>>>>>>>>>>>>>start evaluate......")
        acc = eval_multi_clf(logger=logger,model=model,dev_data_loader=dev_data_loader,device=args.device)
    else:
        logger.info(">>>>>>>>>>>>>>>>>>>start train......")

        global_steps = 0
        for epoch in range(args.num_epochs):

            all_loss_g, all_loss_d, all_loss_c = 0,0,0

            for step, batch in enumerate(train_data_loader, start=1):
                global_steps += 1
                batch_data = [i.to(args.device) for i in batch]

                logits, loss_g, loss_d, loss_c = model(input_ids=batch_data[0], attention_mask=batch_data[1], token_type_ids=batch_data[2],
                                    labels=batch_data[3])
                
                all_loss_g += loss_g.item()
                all_loss_d += loss_d.item()
                all_loss_c += loss_c.item()
                
                
                
                loss = loss_g + loss_d + loss_c
                loss.backward()
                torch.nn.utils.clip_grad_norm_([v for k, v in paras.items()], max_norm=1)
                optimizer.step()
                scheduler.step()
                model.zero_grad()


    
                if global_steps % steps_per_epoch == 0:
                    logger.info("epoch:{},\tloss_g:{},\tloss_d:{},\tloss_c:{}".format(epoch, all_loss_g,all_loss_d,all_loss_c))
                    #! 开始测试！
                    logger.info(">>>>>>>>>>>>>>>>>>>start evaluate......")
                    acc = eval_multi_clf(
                        logger=logger,
                        model=model,
                        dev_data_loader=dev_data_loader,
                        device=args.device)

                    if acc > best_acc:
                        logger.info('save best model to {}'.format(best_model_dir))
                        best_acc = acc
                        model.save(save_dir=best_model_dir)
                        ff_best.write(str(best_acc) + '\n')
                        ff_best.flush()
                    model.save(save_dir=latest_model_dir)
                    ff_test.write(str(acc) + '\n')
                    ff_test.flush()


            train_data_loader.reset(doshuffle=True)

            if args.re_process_train_data and args.train_type == 'one-by-one-in-same-batch':
                with open(args.train_data_path, 'r', encoding='utf-8') as fr:
                    dataset = json.load(fr)
                random.shuffle(dataset)
                examplesss_train = process_dataset(file=dataset, label2id_data_path=args.label2id_path,
                                                max_len=args.train_max_len, lower=True)
                train_data_loader = MWPDatasetLoader(data=examplesss_train, batch_size=args.batch_size, shuffle=False,
                                                    tokenizer=tokenizer, seed=72, sort=False)

    ff_best.close()
    ff_test.close()
