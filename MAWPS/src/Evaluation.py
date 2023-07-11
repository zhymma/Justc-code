import numpy as np
import torch
from tqdm import tqdm
from .MwpDataset import MwpDataSet, process_mwp_V1, process_mwp_V1_plus
from .Models import MwpBertModel, MwpBertModel_CLS
from .verification_labels_plus import re_construct_expression_from_codes, build_expression_by_grous, verification
from .Utils import process_one_mawps_for_test, process_one_mawps_for_test_no_None
from .Test_new import check_codes_acc, check_answer_acc

def eval_multi_clf(model, test_mwps, device, num_labels, test_dev_max_len,label2id_or_value,id2label_or_value,tokenizer,logger=None):
    
    outputs_0 = []
    outputs_1 = []
    num_codes_labels_list = []
    num_positions_list = []
    # pbar = tqdm(test_mwps)
    for idd, raw_mwp in enumerate(test_mwps):
        processed_mwp = process_one_mawps_for_test_no_None(raw_mwp, label2id_or_value, test_dev_max_len, True,
                                                       tokenizer)
        if processed_mwp is not None:
                (sen_tokens, attention_mask, token_type_id, problem_id,num_positions,num_codes_labels) = list(zip(*processed_mwp))
                batch = [torch.tensor(sen_tokens).long(), torch.tensor(attention_mask).long(), torch.tensor(token_type_id).long(),
                         torch.tensor(num_positions).long(),torch.tensor(num_codes_labels).long()]

                model.eval()
                with torch.no_grad():
                    batch_data = [i.to(device) for i in batch]

                    outputs,_ = model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                   token_type_ids=batch_data[2], num_positions = batch_data[3],num_codes_labels=batch_data[4])
                    # outputs = alloutputs[0].squeeze(0)
                    outputs = outputs.squeeze(0)
                    outputs = outputs.to("cpu").numpy()
                    outputs_0.append(outputs)
                    
                    # outputs = alloutputs[1].squeeze(0)
                    # outputs = outputs.to("cpu").numpy()
                    # outputs_1.append(outputs)
                    
                    num_codes_labels = batch_data[4].reshape(-1,28)
                    num_codes_labels_list.append(num_codes_labels)
                    num_positions = batch_data[3].reshape(-1,1).clone()
                    num_positions[:,0] -= 1
                    num_positions_list.append(num_positions)
                      
    right_codes_count = 0
    right_ans_count = 0
    wrong_ans_count = 0
    wrong_be_tree = 0
    for i,raw_mwp in enumerate(test_mwps):
        labels, all_logits = [], []
        labels_pos = []
        outputs = outputs_0[i]
        num_codes_labels = num_codes_labels_list[i]
        num_positions = num_positions_list[i]
        new_label_pos = torch.cat((num_positions,num_codes_labels),dim =1)
        labels.append(num_codes_labels.to("cpu").numpy())
        labels_pos.append(new_label_pos.to("cpu").numpy())
        all_logits.append(outputs)
        labels = np.vstack(labels)
        all_logits = np.vstack(all_logits)
        labels_pos = np.vstack(labels_pos)
        num_codes_labels = num_codes_labels.to("cpu").numpy()
        # if check_codes_acc(raw_mwp, labels, all_logits):
        #     right_codes_count += 1
        #! answer acc
        if check_answer_acc(raw_mwp, labels_pos, all_logits, id2label_or_value):
            right_ans_count += 1
        else:
            # false_mwp.append(raw_mwp)
            wrong_ans_count += 1
            if raw_mwp['pre_final_expression'] == 'Failed':
                wrong_be_tree +=1
        #! 计算code acc code check acc
        right = True
        for i in range(outputs.shape[0]):
            A = outputs[i,:]
            B = num_codes_labels[i,:]
            dd_pred = np.array([round(i) for i in A])
            dd_label = np.array([round(i) for i in B])

            if not (dd_pred == dd_label).all() :
                right = False     
        if right:
            right_codes_count += 1

    ans_acc = right_ans_count / len(test_mwps)
    code_acc = right_codes_count/len(test_mwps)
    if logger is not None:
        logger.info('right_count:{}\ttotal:{}\t Answer ACC: {}'.format(right_ans_count, len(test_mwps), ans_acc))
        logger.info('right_codes_count:{}\ttotal:{}\t Code ACC: {}'.format(right_codes_count, len(test_mwps), code_acc))
        logger.info('wrong_be_tree_count:{}\twrong_total:{}\t wrong be tree ACC: {}'.format(wrong_be_tree, wrong_ans_count, wrong_be_tree/wrong_ans_count))
        
    #! 第二次迭代结果输出
    # logger.info('\n')

    # right_codes_count = 0
    # right_ans_count = 0
    # wrong_ans_count = 0
    # wrong_be_tree = 0
    # for i,raw_mwp in enumerate(test_mwps):
    #     labels, all_logits = [], []
    #     labels_pos = []
    #     outputs = outputs_1[i]
    #     num_codes_labels = num_codes_labels_list[i]
    #     num_positions = num_positions_list[i]
    #     new_label_pos = torch.cat((num_positions,num_codes_labels),dim =1)
    #     labels.append(num_codes_labels.to("cpu").numpy())
    #     labels_pos.append(new_label_pos.to("cpu").numpy())
    #     all_logits.append(outputs)
    #     labels = np.vstack(labels)
    #     all_logits = np.vstack(all_logits)
    #     labels_pos = np.vstack(labels_pos)
    #     num_codes_labels = num_codes_labels.to("cpu").numpy()
    #     # if check_codes_acc(raw_mwp, labels, all_logits):
    #     #     right_codes_count += 1
    #     #! answer acc
    #     if check_answer_acc(raw_mwp, labels_pos, all_logits, id2label_or_value):
    #         right_ans_count += 1
    #     else:
    #         # false_mwp.append(raw_mwp)
    #         wrong_ans_count += 1
    #         if raw_mwp['pre_final_expression'] == 'Failed':
    #             wrong_be_tree +=1
    #     #! 计算code acc code check acc
    #     right = True
    #     for i in range(outputs.shape[0]):
    #         A = outputs[i,:]
    #         B = num_codes_labels[i,:]
    #         dd_pred = np.array([round(i) for i in A])
    #         dd_label = np.array([round(i) for i in B])

    #         if not (dd_pred == dd_label).all() :
    #             right = False     
    #     if right:
    #         right_codes_count += 1

    # ans_acc = right_ans_count / len(test_mwps)
    # code_acc = right_codes_count/len(test_mwps)
    # if logger is not None:
    #     logger.info('right_count:{}\ttotal:{}\t Answer ACC: {}'.format(right_ans_count, len(test_mwps), ans_acc))
    #     logger.info('right_codes_count:{}\ttotal:{}\t Code ACC: {}'.format(right_codes_count, len(test_mwps), code_acc))
    #     logger.info('wrong_be_tree_count:{}\twrong_total:{}\t wrong be tree ACC: {}'.format(wrong_be_tree, wrong_ans_count, wrong_be_tree/wrong_ans_count))
    
    model.train()
    return ans_acc

def eval_multi_clf1(model, test_mwps, device, num_labels, test_dev_max_len,label2id_or_value,id2label_or_value,tokenizer,logger=None):
    
    right_codes_count = 0
    right_ans_count = 0
    wrong_ans_count = 0
    wrong_be_tree = 0
    code_check_right = 0
    code_check_right_0 = 0
    code_check_right_1 = 0
    code_check_total = 0
    code_check_1 = 0
    # pbar = tqdm(test_mwps)
    for idd, raw_mwp in enumerate(test_mwps):
        processed_mwp = process_one_mawps_for_test_no_None(raw_mwp, label2id_or_value, test_dev_max_len, True,
                                                       tokenizer)
        if processed_mwp is not None:
                (sen_tokens, attention_mask, token_type_id, problem_id,num_positions,num_codes_labels) = list(zip(*processed_mwp))
                batch = [torch.tensor(sen_tokens).long(), torch.tensor(attention_mask).long(), torch.tensor(token_type_id).long(),
                         torch.tensor(num_positions).long(),torch.tensor(num_codes_labels).long()]

                model.eval()
                labels, all_logits = [], []
                labels_pos = []
                with torch.no_grad():
                    batch_data = [i.to(device) for i in batch]

                    outputs, code_check_pred,iter_num = model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                   token_type_ids=batch_data[2], num_positions = batch_data[3],num_codes_labels=batch_data[4])
                    outputs = outputs.squeeze(0)
                    outputs = outputs.to("cpu").numpy()
                    
                    num_codes_labels = batch_data[4].reshape(-1,28)
                    num_positions = batch_data[3].reshape(-1,1)
                    num_positions[:,0] -= 1
                    new_label_pos = torch.cat((num_positions,num_codes_labels),dim =1)
            
                    labels.append(num_codes_labels.to("cpu").numpy())
                    labels_pos.append(new_label_pos.to("cpu").numpy())
                    all_logits.append(outputs)

                labels = np.vstack(labels)
                all_logits = np.vstack(all_logits)
                labels_pos = np.vstack(labels_pos)
                code_check_pred = code_check_pred.to("cpu").numpy()
                code_check_pred = code_check_pred[0]
                num_codes_labels = num_codes_labels.to("cpu").numpy()
                # if check_codes_acc(raw_mwp, labels, all_logits):
                #     right_codes_count += 1
                #! answer acc
                if check_answer_acc(raw_mwp, labels_pos, all_logits, id2label_or_value):
                    right_ans_count += 1
                else:
                    # false_mwp.append(raw_mwp)
                    wrong_ans_count += 1
                    if raw_mwp['pre_final_expression'] == 'Failed':
                        wrong_be_tree +=1
                #! 计算code acc code check acc
                right = True
                for i in range(outputs.shape[0]):
                    A = outputs[i,:]
                    B = num_codes_labels[i,:]
                    dd_pred = np.array([round(i) for i in A])
                    dd_label = np.array([round(i) for i in B])
                    if code_check_pred[i].argmax() == 1:
                        code_check_1 += 1
                    if not (dd_pred == dd_label).all() :
                        right = False
                        if code_check_pred[i].argmax() == 0:
                            code_check_right_0+=1
                    else :
                        if code_check_pred[i].argmax() == 1:
                            code_check_right_1+=1
                    code_check_total+=1
                if right:
                    right_codes_count += 1
    
    
    code_check_right = code_check_right_0 + code_check_right_1
    ans_acc = right_ans_count / len(test_mwps)
    code_acc = right_codes_count/len(test_mwps)
    if logger is not None:
        logger.info('right_count:{}\ttotal:{}\t Answer ACC: {}'.format(right_ans_count, len(test_mwps), ans_acc))
        logger.info('right_codes_count:{}\ttotal:{}\t Code ACC: {}'.format(right_codes_count, len(test_mwps), code_acc))
        logger.info('code_check_right: {}\tcode_check_right0: {}\tcode_check_right1: {}\tcode_check_total: {}\t code check acc:{}'.format(code_check_right,code_check_right_0,code_check_right_1, code_check_total, code_check_right/code_check_total))
        logger.info('wrong_be_tree_count:{}\twrong_total:{}\t wrong be tree ACC: {}'.format(wrong_be_tree, wrong_ans_count, wrong_be_tree/wrong_ans_count))
        logger.info('code_check_1: {}\tcode_check_total: {}\t code check 1:{}'.format(code_check_1, code_check_total, code_check_1/code_check_total))
        
    model.train()
    return ans_acc

def eval_multi_clf0(model, dev_data_loader, device, num_labels, logger=None):
    model.eval()
    count = 0
    total_len = 0 
    code_check_right = 0
    code_check_total = 0
    with torch.no_grad():
        for batch in dev_data_loader:
            batch_data = [i.to(device) for i in batch]
            input_ids, input_mask, token_type_ids, problem_id, num_positions, num_codes_labels = batch_data
            outputs, code_check_pred = model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=token_type_ids,
                                 num_positions=num_positions, num_codes_labels=num_codes_labels)
            
            outputs = outputs.squeeze(0)
            outputs = outputs.to("cpu").numpy()
            num_codes_labels = num_codes_labels.to("cpu").numpy()
            code_check_pred = code_check_pred.to("cpu").numpy()
            num_codes_labels = num_codes_labels[0].reshape(-1,num_labels)
            code_check_pred = code_check_pred[0]
            #! 判断两个是否相等
            right = True
            for i in range(outputs.shape[0]):
                A = outputs[i,:]
                B = num_codes_labels[i,:]
                dd_pred = np.array([round(i) for i in A])
                dd_label = np.array([round(i) for i in B])
                if not (dd_pred == dd_label).all() :
                    right = False
                    if code_check_pred[i].argmax() == 0:
                        code_check_right+=1
                else :
                    if code_check_pred[i].argmax() == 1:
                        code_check_right+=1
                code_check_total+=1
            total_len += 1
            if right:
                count += 1
    if logger is not None:
        logger.info('right: {}\ttotal: {}\tM-tree codes acc: {} code check acc:{}'.format(count, total_len, count / total_len, code_check_right/code_check_total))

    model.train()
    return count / total_len

def eval_multi_clf_for_test_new(model, test_mwps, device, num_labels, test_dev_max_len,label2id_or_value,id2label_or_value,tokenizer,logger=None):
    
    outputs_0 = []
    num_codes_labels_list = []
    num_positions_list = []
    for idd, raw_mwp in enumerate(test_mwps):
        processed_mwp = process_one_mawps_for_test_no_None(raw_mwp, label2id_or_value, test_dev_max_len, True,
                                                       tokenizer)
        if processed_mwp is not None:
                (sen_tokens, attention_mask, token_type_id, problem_id,num_positions,num_codes_labels) = list(zip(*processed_mwp))
                batch = [torch.tensor(sen_tokens).long(), torch.tensor(attention_mask).long(), torch.tensor(token_type_id).long(),
                         torch.tensor(num_positions).long(),torch.tensor(num_codes_labels).long()]

                model.eval()
                with torch.no_grad():
                    batch_data = [i.to(device) for i in batch]

                    outputs,_ = model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                   token_type_ids=batch_data[2], num_positions = batch_data[3],num_codes_labels=batch_data[4])
                    outputs = outputs.squeeze(0)
                    outputs = outputs.to("cpu").numpy()
                    outputs_0.append(outputs)
                    
                    num_codes_labels = batch_data[4].reshape(-1,28)
                    num_codes_labels_list.append(num_codes_labels)
                    num_positions = batch_data[3].reshape(-1,1).clone()
                    num_positions[:,0] -= 1
                    num_positions_list.append(num_positions)
                      
    right_codes_count = 0
    all_codes_count = 0
    right_ans_count = 0
    wrong_ans_count = 0
    wrong_be_tree = 0
    temp = 0
    temp1 = 0
    for i,raw_mwp in enumerate(test_mwps):
        labels, all_logits = [], []
        labels_pos = []
        outputs = outputs_0[i]
        num_codes_labels = num_codes_labels_list[i]
        num_positions = num_positions_list[i]
        new_label_pos = torch.cat((num_positions,num_codes_labels),dim =1)
        labels.append(num_codes_labels.to("cpu").numpy())
        labels_pos.append(new_label_pos.to("cpu").numpy())
        all_logits.append(outputs)
        labels = np.vstack(labels)
        all_logits = np.vstack(all_logits)
        labels_pos = np.vstack(labels_pos)
        num_codes_labels = num_codes_labels.to("cpu").numpy()
        # if check_codes_acc(raw_mwp, labels, all_logits):
        #     right_codes_count += 1
        #! answer acc
        if check_answer_acc(raw_mwp, labels_pos, all_logits, id2label_or_value):
            right_ans_count += 1
        else:
            # false_mwp.append(raw_mwp)
            wrong_ans_count += 1
            if raw_mwp['pre_final_expression'] == 'Failed':
                wrong_be_tree +=1
        #! 计算code acc code check acc
        
        #! temp 找到最小的大于0.5的值，判断它是不是错误的
        min_value = np.min(outputs[outputs > 0.5])
        indices = np.where(outputs == min_value)# 希望没有两个一样的最小值
        if(num_codes_labels[indices] == np.array([0])):
            temp += 1
        max_value = np.max(outputs[outputs < 0.5])
        indices = np.where(outputs == max_value)
        if(num_codes_labels[indices] != np.array([0])):
            temp1 += 1
        
        for i in range(outputs.shape[0]):
            A = outputs[i,:]
            B = num_codes_labels[i,:]
            dd_pred = np.array([round(i) for i in A]).astype(np.int32)
            dd_label = np.array([round(i) for i in B]).astype(np.int32)

            if  (dd_pred == dd_label).all() :
                right_codes_count += 1
            else:
                logger.info(A)
                logger.info(dd_pred)
                logger.info(dd_label)
                logger.info("\n")
            all_codes_count += 1


    ans_acc = right_ans_count / len(test_mwps)
    code_acc = right_codes_count/all_codes_count
    if logger is not None:
        logger.info('right_count:{}\ttotal:{}\t Answer ACC: {}'.format(right_ans_count, len(test_mwps), ans_acc))
        logger.info('right_codes_count:{}\ttotal:{}\tCode ACC: {}\twrong_be_tree_count:{}\twrong_total:{}\t wrong be tree ACC: {}'.format(right_codes_count, all_codes_count, code_acc,wrong_be_tree, wrong_ans_count, wrong_be_tree/wrong_ans_count))
        logger.info('temp:{}\ttemp1:{}\twrong_total:{}\t'.format(temp,temp1,all_codes_count-right_codes_count ))
    model.train()
    return ans_acc


def eval_multi_clf_for_classfier(model, test_mwps, device, num_labels, test_dev_max_len,label2id_or_value,id2label_or_value,tokenizer,logger=None):
    
    outputs_0 = []
    num_codes_labels_list = []
    num_positions_list = []
    for idd, raw_mwp in enumerate(test_mwps):
        processed_mwp = process_one_mawps_for_test_no_None(raw_mwp, label2id_or_value, test_dev_max_len, True,
                                                       tokenizer)
        if processed_mwp is not None:
                (sen_tokens, attention_mask, token_type_id, problem_id,num_positions,num_codes_labels) = list(zip(*processed_mwp))
                batch = [torch.tensor(sen_tokens).long(), torch.tensor(attention_mask).long(), torch.tensor(token_type_id).long(),
                         torch.tensor(num_positions).long(),torch.tensor(num_codes_labels).long()]
                model.eval()
                with torch.no_grad():
                    batch_data = [i.to(device) for i in batch]

                    outputs,_ = model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                   token_type_ids=batch_data[2], num_positions = batch_data[3],num_codes_labels=batch_data[4])
                    # outputs = alloutputs[0].squeeze(0)
                    outputs = outputs.squeeze(0)
                    outputs = outputs.to("cpu").numpy()
                    outputs_0.append(outputs)

                    num_codes_labels = batch_data[4].reshape(-1,28)
                    num_codes_labels_list.append(num_codes_labels)
                    num_positions = batch_data[3].reshape(-1,1).clone()
                    num_positions[:,0] -= 1
                    num_positions_list.append(num_positions)
                      
    right_codes_count = 0
    all_codes_count = 0
    right_ans_count = 0
    wrong_ans_count = 0
    wrong_be_tree = 0
    temp = 0
    temp1 = 0
    for i,raw_mwp in enumerate(test_mwps):
        labels, all_logits = [], []
        labels_pos = []
        outputs = outputs_0[i]
        num_codes_labels = num_codes_labels_list[i]
        num_positions = num_positions_list[i]
        new_label_pos = torch.cat((num_positions,num_codes_labels),dim =1)
        num_codes_labels = num_codes_labels.to("cpu").numpy()

        
        
        labels.append(num_codes_labels)
        labels_pos.append(new_label_pos.to("cpu").numpy())
        all_logits.append(outputs)
        labels = np.vstack(labels)
        all_logits = np.vstack(all_logits)
        labels_pos = np.vstack(labels_pos)
        
        #! answer acc
        if check_answer_acc(raw_mwp, labels_pos, all_logits, id2label_or_value):
            right_ans_count += 1
        else:
            # false_mwp.append(raw_mwp)
            wrong_ans_count += 1
            if raw_mwp['pre_final_expression'] == 'Failed':
                wrong_be_tree +=1
        #! 计算code acc code check acc

        #! temp 找到最小的大于0.5的值，判断它是不是错误的
        min_value = np.min(outputs[outputs > 0.5])
        indices = np.where(outputs == min_value)# 希望没有两个一样的最小值
        if(num_codes_labels[indices] == np.array([0])):
            temp += 1
        max_value = np.max(outputs[outputs < 0.5])
        indices = np.where(outputs == max_value)
        if(num_codes_labels[indices] != np.array([0])):
            temp1 += 1

        for i in range(outputs.shape[0]):
            A = outputs[i,:]
            B = num_codes_labels[i,:]
            dd_pred = np.array([round(i) for i in A]).astype(np.int32)
            dd_label = np.array([round(i) for i in B]).astype(np.int32)

            if  (dd_pred == dd_label).all() :
                right_codes_count += 1
            # else:
            #     logger.info(A)
            #     logger.info(dd_pred)
            #     logger.info(dd_label)
            #     logger.info("\n")
            all_codes_count += 1


    ans_acc = right_ans_count / len(test_mwps)
    code_acc = right_codes_count/all_codes_count
    if logger is not None:
        logger.info('right_count:{}\ttotal:{}\t Answer ACC: {}'.format(right_ans_count, len(test_mwps), ans_acc))
        logger.info('right_codes_count:{}\ttotal:{}\tCode ACC: {}\twrong_be_tree_count:{}\twrong_total:{}\t wrong be tree ACC: {}'.format(right_codes_count, all_codes_count, code_acc,wrong_be_tree, wrong_ans_count, wrong_be_tree/wrong_ans_count))
        logger.info('temp:{}\ttemp1:{}\twrong_total:{}\t'.format(temp,temp1,len(test_mwps)-right_ans_count ))
        
    model.train()
    return ans_acc


def eval_multi_clf_for_classfier_check(model, test_mwps, device, num_labels, test_dev_max_len,label2id_or_value,id2label_or_value,tokenizer,refiner,logger=None):
    
    outputs_0 = []
    num_codes_labels_list = []
    num_positions_list = []


    right_checker = 0


    for idd, raw_mwp in enumerate(test_mwps):
        processed_mwp = process_one_mawps_for_test_no_None(raw_mwp, label2id_or_value, test_dev_max_len, True,
                                                       tokenizer)
        if processed_mwp is not None:
                (sen_tokens, attention_mask, token_type_id, problem_id,num_positions,num_codes_labels) = list(zip(*processed_mwp))
                batch = [torch.tensor(sen_tokens).long(), torch.tensor(attention_mask).long(), torch.tensor(token_type_id).long(),
                         torch.tensor(num_positions).long(),torch.tensor(num_codes_labels).long()]
                model.eval()
                with torch.no_grad():
                    batch_data = [i.to(device) for i in batch]

                    outputs,p_hidden = model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                                   token_type_ids=batch_data[2], num_positions = batch_data[3],num_codes_labels=batch_data[4])

                    _, refiner_outputs ,check_labels=refiner(outputs, batch_data[4].clone(),p_hidden)

                    #! 统计checker的准确率
                    # equality = torch.eq(torch.round(refiner_outputs),check_labels)
                    # if(equality.all()):
                    #     right_checker += 1
                    #! 统计corrector的准确率
                    refiner_outputs = refiner_outputs.to("cpu").numpy()
                    refiner_outputs  = refiner_outputs.squeeze(0)
                    # outputs = outputs.squeeze(0)
                    outputs = outputs.to("cpu").numpy()
                    outputs_0.append(refiner_outputs)
                    # outputs_0.append(outputs)

                    num_codes_labels = batch_data[4].reshape(-1,28)
                    num_codes_labels_list.append(num_codes_labels)
                    num_positions = batch_data[3].reshape(-1,1).clone()
                    num_positions[:,0] -= 1
                    num_positions_list.append(num_positions)
                    

                      
    right_codes_count = 0
    all_codes_count = 0
    right_ans_count = 0
    wrong_ans_count = 0
    wrong_be_tree = 0
    temp = 0
    temp1 = 0
    for i,raw_mwp in enumerate(test_mwps):
        labels, all_logits = [], []
        labels_pos = []
        outputs = outputs_0[i]
        num_codes_labels = num_codes_labels_list[i]
        num_positions = num_positions_list[i]
        new_label_pos = torch.cat((num_positions,num_codes_labels),dim =1)
        num_codes_labels = num_codes_labels.to("cpu").numpy()


            
        labels.append(num_codes_labels)
        labels_pos.append(new_label_pos.to("cpu").numpy())
        all_logits.append(outputs)
        labels = np.vstack(labels)
        all_logits = np.vstack(all_logits)
        labels_pos = np.vstack(labels_pos)
        
        #! answer acc
        if check_answer_acc(raw_mwp, labels_pos, all_logits, id2label_or_value):
            right_ans_count += 1
        else:
            # false_mwp.append(raw_mwp)
            wrong_ans_count += 1
            if raw_mwp['pre_final_expression'] == 'Failed':
                wrong_be_tree +=1
        #! 计算code acc code check acc

        #! temp 找到最小的大于0.5的值，判断它是不是错误的
        
        min_value = np.min(outputs[outputs > 0.5])
        indices = np.where(outputs == min_value)# 希望没有两个一样的最小值
        if(num_codes_labels[indices] == np.array([0])):
            temp += 1
        max_value = np.max(outputs[outputs < 0.5])
        indices = np.where(outputs == max_value)
        if(num_codes_labels[indices] != np.array([0])):
            temp1 += 1
        right = True
        for i in range(outputs.shape[0]):
            A = outputs[i,:]
            B = num_codes_labels[i,:]
            dd_pred = np.array([round(i) for i in A]).astype(np.int32)
            dd_label = np.array([round(i) for i in B]).astype(np.int32)

            if  (dd_pred == dd_label).all() :
                right_codes_count += 1
            all_codes_count += 1

    ans_acc = right_ans_count / len(test_mwps)
    code_acc = right_codes_count/all_codes_count
    if logger is not None:
        logger.info('right_count:{}\ttotal:{}\t Answer ACC: {}'.format(right_ans_count, len(test_mwps), ans_acc))
        logger.info('right_codes_count:{}\ttotal:{}\tCode ACC: {}\twrong_be_tree_count:{}\twrong_total:{}\t wrong be tree ACC: {}'.format(right_codes_count, all_codes_count, code_acc,wrong_be_tree, wrong_ans_count, wrong_be_tree/wrong_ans_count))
        logger.info('temp:{}\ttemp1:{}\twrong_total:{}\t'.format(temp,temp1,len(test_mwps)-right_ans_count ))
    
    return ans_acc

    # print(right_checker,len(test_mwps),right_checker/len(test_mwps))
    # return right_checker/len(test_mwps)
