
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
from src.Test_new import *

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def train(args):

    setup_seed(args.seed)
    # # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device    
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
    false_data_dir = os.path.join(args.output_dir, "false_data")
    os.makedirs(false_data_dir, exist_ok=True)

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
        
        with open(args.train_data_path, 'r', encoding='utf-8')as ffs:
            train_mwps = json.load(ffs)
        with open(args.dev_data_path, 'r', encoding='utf-8')as ffs:
            test_mwps = json.load(ffs)
    else:
        print('args.train_type == together not yet!!!')
        sys.exit()

    # model
    logger.info("define model...")


    model = UTSC_Solver(bert_path=args.pretrain_model_path, num_labels=num_labels, iter_num=args.iter_num,num_layers=args.transformer_layes_num,hidden_size=args.fc_hidden_size)

    # model.load_state_dict(torch.load("./output/test/best_model/model.pt"))

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
            "params": [p for n, p in paras.items() if not any(nd in n for nd in no_decay) ],
            "weight_decay": 0.01,
        }, {
            "params": [p for n, p in paras.items() if any(nd in n for nd in no_decay) ],
            "weight_decay": 0.0
        }
    ]

    total_steps = int(len(train_data_loader) * args.num_epochs)
    if args.warmup < 1:
        warmup_steps = int(total_steps * args.warmup)
    else:
        warmup_steps = int(args.warmup)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr,no_deprecation_warning=True)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,num_training_steps=total_steps)
    # 推荐RMSProp，SGD也行。Adam不行
    disc_optimizer = torch.optim.RMSprop(model.discriminator.parameters(), lr=0.0001)
    best_acc = -1
    if args.mode == "train":
        #  start training
        
        logger.info("\n>>>>>>>>>>>>>>>>>>>start train......")

        for epoch in range(args.num_epochs):
            all_loss_g = 0
            all_loss_d = 0
            loss = 0
            code_right = torch.zeros(args.iter_num).cuda()
            code_all = torch.zeros(args.iter_num).cuda()
            judgement_right = torch.zeros(args.iter_num).cuda()
            judgement_all = torch.zeros(args.iter_num).cuda()
            model.train()

            for step, batch in enumerate(train_data_loader, start=1):
                batch_data = [i.to(args.device) for i in batch]
                # (input_ids, input_mask, token_type_ids, problem_id, num_positions, tgt_ids, tgt_mask)
                # input_mask和tgt_mask前面是1后面是0
                # token_type_ids的数值为1，其余为0
                input_ids, input_mask, token_type_ids, problem_id, num_positions, tgt_ids, tgt_mask = batch_data
                if epoch < 6: #!单独训练generator
                    for param in model.discriminator.parameters():
                        param.requires_grad = False
                    loss_g, loss_d, code_pred_list, judgement_pred_list, discriminator_label_list = model(input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids, num_positions=num_positions, tgt_ids=tgt_ids, tgt_mask=tgt_mask,problem_id=problem_id,train_type = 0)
                    all_loss_g += loss_g.item()
                    all_loss_d += loss_d.item()
                    loss = loss_g
                    # loss = loss_d
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([v for k, v in paras.items()], max_norm=1)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()

                else: #!训练discriminator
                    #便利所有参数，将除了discriminator的所有requires_grad设置为False
                    for param in model.parameters():
                        param.requires_grad = False
                    for param in model.discriminator.parameters():
                        param.requires_grad = True

                    loss_g, loss_d, code_pred_list, judgement_pred_list, discriminator_label_list = model(input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids, num_positions=num_positions, tgt_ids=tgt_ids, tgt_mask=tgt_mask,problem_id=problem_id,train_type = 0)
                    all_loss_g += loss_g.item()
                    all_loss_d += loss_d.item()
                    # loss = 0.1 * loss_g + loss_d
                    loss = loss_d
                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_([v for k, v in paras.items()], max_norm=1)
                    # optimizer.step()
                    # scheduler.step()
                    disc_optimizer.step()
                    model.zero_grad()

                # ! 评测指标 code_acc judgement_acc 优化下面的for循环
                tgt_mask_sum = torch.sum(tgt_mask)

                for i in range(args.iter_num):
                    #Todo 计算ans_acc
                    # 计算discriminator_label_list[i]在tgt_mask的位置的和
                    code_right[i] += torch.sum((discriminator_label_list[i]) * tgt_mask)
                    code_all[i] += tgt_mask_sum

                    # 计算judgement_pred_list[i] == discriminator_label_list[i]在tgt_mask的位置的和
                    judgement_right[i] += torch.sum((judgement_pred_list[i] == discriminator_label_list[i]) * tgt_mask)
                    judgement_all[i] += tgt_mask_sum

            logger.info("---------------------------------------------------------------")
            logger.info("epoch:{},\tloss_g:{}\tloss_d:{}".format(epoch, all_loss_g, all_loss_d))
            logger.info("train_code_acc:{}".format([code_right[i].item() / code_all[i].item() for i in range(args.iter_num)]))
            logger.info("train_judgement_acc:{}".format([judgement_right[i].item() / judgement_all[i].item() for i in range(args.iter_num)]))

            # acc = eval_utsc_solver(
            # logger=logger,
            # model=model,
            # test_mwps=test_mwps,
            # dev_data_loader=dev_data_loader,
            # device=args.device,
            # id2label_or_value = id2label_or_value,
            # )
            acc = eval_utsc_solver_iterations(
            logger=logger,
            model=model,
            test_mwps=test_mwps,
            dev_data_loader=dev_data_loader,
            device=args.device,
            iter_num = args.iter_num,
            id2label_or_value = id2label_or_value,
            )

            if acc >= best_acc:
                logger.info('save best model to {}, best acc is:{} '.format(best_model_dir, acc))
                best_acc = acc
                # 保存模型
                model.eval()
                torch.save(model.state_dict(), best_model_dir+"/model.pt")


            train_data_loader.reset(doshuffle=True)

        #测试最终结果
        logger.info("\n\n")
        logger.info("Final_test")
        model.load_state_dict(torch.load(best_model_dir+"/model.pt"))
        model.to(args.device)
        acc = eval_utsc_solver_iterations(
            logger=logger,
            model=model,
            test_mwps=test_mwps,
            dev_data_loader=dev_data_loader,
            device=args.device,
            iter_num = args.iter_num,
            id2label_or_value = id2label_or_value,
            )


    else:
        #测试最终结果

        model.load_state_dict(torch.load(best_model_dir+"/model.pt"))
        model.to(args.device)

        logger1 = logging.getLogger()
        logger1.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger1.addHandler(console_handler)
        logger1.info("\n\nTest mode!\n")
        acc = eval_utsc_solver_iterations(
            logger=logger1,
            model=model,
            test_mwps=test_mwps,
            dev_data_loader=dev_data_loader,
            device=args.device,
            iter_num = args.iter_num,
            id2label_or_value = id2label_or_value,
            test_mode=True,
            json_path = false_data_dir
            )
    
    