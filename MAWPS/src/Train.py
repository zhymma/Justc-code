
import json
import logging
import math
import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup,get_cosine_schedule_with_warmup
from src.Utils import process_dataset, MWPDatasetLoader
from src.Models import *
from src.Evaluation import *


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def train(args):
    # devices setting
    #! 设置随机数种子
    setup_seed(args.seed)
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
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
    handler = logging.FileHandler(os.path.join(log_dir, args.log_file))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.addHandler(console)

    logger.info("\n\n")

    
    with open(args.label2id_path, 'r', encoding='utf-8') as f:
        label2id = json.load(f)
    num_labels = len(label2id)
    label2id_or_value = {}
    id2label_or_value = {}
    with open(args.label2id_path, 'r', encoding='utf-8')as ff:
        label2id_list = json.load(ff)
    for inde, label in enumerate(label2id_list):
        label2id_or_value[label] = inde
        id2label_or_value[str(inde)] = label
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model_path)

    if args.train_type != 'together':
        # train data...
        logger.info("get train data loader...")
        examplesss_train = process_dataset(file=args.train_data_path, label2id_data_path=args.label2id_path,
                                           max_len=args.train_max_len, lower=True)

        if args.train_type == 'one-by-one-in-same-batch':
            # one by one 的话 shuffle 洗牌 和 按长度降序排序 都要为 False
            train_data_loader = MWPDatasetLoader(data=examplesss_train, batch_size=args.batch_size, shuffle=False,
                                                 tokenizer=tokenizer, sort=False)
        elif args.train_type == 'one-by-one-random':
            train_data_loader = MWPDatasetLoader(data=examplesss_train, batch_size=args.batch_size, shuffle=True,
                                                 tokenizer=tokenizer, sort=False)
        else:
            print('args.train_type wrong!!!')
            sys.exit()
        # dev data...
        logger.info("get dev data loader...")
        examplesss_test = process_dataset(file=args.dev_data_path, label2id_data_path=args.label2id_path,
                                          max_len=args.test_dev_max_len, lower=True)
        dev_data_loader = MWPDatasetLoader(data=examplesss_test, batch_size=1, shuffle=False,
                                           tokenizer=tokenizer, sort=False)
        with open(args.dev_data_path, 'r', encoding='utf-8')as ffs:
            test_mwps = json.load(ffs)
    else:
        print('args.train_type == together not yet!!!')
        sys.exit()

    

    # model
    logger.info("define model...")


    model = UTSC_Solver(bert_path=args.pretrain_model_path, num_labels=num_labels, iter_num=args.iter_num,hidden_size=args.fc_hidden_size)


    model.to(args.device)
    model.zero_grad()
    model.train()

    # optimizer
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

    total_steps = int(len(train_data_loader) * args.num_epochs)

    if args.warmup < 1:
        warmup_steps = int(total_steps * args.warmup)
    else:
        warmup_steps = int(args.warmup)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,no_deprecation_warning=True)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        else:
            return 0.5 * (1 + math.cos((step - warmup_steps) / (total_steps-warmup_steps) * math.pi))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)
    
    best_acc = -1
    if args.mode == "train":
        #  start training
        
        logger.info("\n>>>>>>>>>>>>>>>>>>>start train......")

        for epoch in range(args.num_epochs):
            all_loss_g = 0
            all_loss_d = 0
            loss = 0
            for step, batch in enumerate(train_data_loader, start=1):
                batch_data = [i.to(args.device) for i in batch]
                # (input_ids, input_mask, token_type_ids, problem_id, num_positions, tgt_ids, tgt_mask)
                input_ids, input_mask, token_type_ids, problem_id, num_positions, tgt_ids, tgt_mask = batch_data
                
                loss_g, loss_d = model(input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids, num_positions=num_positions, tgt_ids=tgt_ids, tgt_mask=tgt_mask,problem_id=problem_id)
                all_loss_g += loss_g.item()
                all_loss_d += loss_d.item()
                loss = loss_g + loss_d
                loss.backward()
                torch.nn.utils.clip_grad_norm_([v for k, v in paras.items()], max_norm=1)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                


            # print loss...
            logger.info("\n")

            logger.info("epoch:{},\tloss_g:{}\tloss_d:{}".format(epoch, all_loss_g, all_loss_d))

            acc = eval_utsc_solver(
            logger=logger,
            model=model,
            test_mwps=test_mwps,
            dev_data_loader=dev_data_loader,
            device=args.device,
            id2label_or_value = id2label_or_value,
            )

            # if acc >= best_acc:
            #     logger.info('save best model to {}'.format(best_model_dir))
            #     best_acc = acc
            #     model.save(save_dir=best_model_dir)
            #     refiner.save(save_dir=best_model_dir)
            # model.save(save_dir=latest_model_dir)
            # refiner.save(save_dir=latest_model_dir)
            train_data_loader.reset(doshuffle=True)

        #测试最终结果
        # logger.info("\n\n")
        # logger.info("final_test")
        # model.load(best_model_dir)
        # model.to(args.device)
        # acc = eval_multi_clf_for_classfier_check(
        #         logger=logger,
        #         model=model,
        #         test_mwps=test_mwps,
        #         device=args.device,
        #         num_labels = num_labels,
        #         test_dev_max_len = args.test_dev_max_len,
        #         label2id_or_value = label2id_or_value,
        #         id2label_or_value = id2label_or_value,
        #         tokenizer = tokenizer,
        #         refiner = refiner
        #         )
        # print("final_test")
        # print(f"Answer acc:{acc}")
        # print("\n")


    else:
        #测试最终结果

        model.load(best_model_dir)
        model.to(args.device)

        logger1 = logging.getLogger()
        logger1.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger1.addHandler(console_handler)

        logger1.info("Test!")
        # acc = eval_multi_clf_for_classfier_check(
        #         logger=logger1,
        #         model=model,
        #         test_mwps=test_mwps,
        #         device=args.device,
        #         num_labels = num_labels,
        #         test_dev_max_len = args.test_dev_max_len,
        #         label2id_or_value = label2id_or_value,
        #         id2label_or_value = id2label_or_value,
        #         tokenizer = tokenizer,
        #         refiner = refiner
        #         )

        # acc = eval_multi_clf_for_test_new(
        #         logger=logger1,
        #         model=model,
        #         test_mwps=test_mwps,
        #         device=args.device,
        #         num_labels = num_labels,
        #         test_dev_max_len = args.test_dev_max_len,
        #         label2id_or_value = label2id_or_value,
        #         id2label_or_value = id2label_or_value,
        #         tokenizer = tokenizer,
        #         json_path = args.output_dir 
        #         )
    
    