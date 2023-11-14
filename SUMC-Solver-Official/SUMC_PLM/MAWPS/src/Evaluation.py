import numpy as np
import torch


def eval_multi_clf(model, dev_data_loader, device, logger=None, T=1.0):
    model.eval()

    disc_right,disc_all = 0,0
    code_right,code_all = 0,0
    TP,TN,FN,FP =0,0 ,0,0
    corrector_right,corrector_all = 0,0
    correct_all_pred_right,correct_all_pred_all = 0,0
    final_code_right,final_code_all = 0,0

    with torch.no_grad():
        for batch in dev_data_loader:
            batch_data = [i.to(device) for i in batch]
            code_pred, loss_value, discriminator_pred, discriminator_label, corrector_pred = model(input_ids=batch_data[0], attention_mask=batch_data[1],
                                       token_type_ids=batch_data[2], labels=batch_data[3])
            #! 计算生成器的准确率
            code_label = batch_data[3][:, 1:]
            code_right += (code_pred == code_label).all(dim=-1).float().sum().item()
            code_all += code_label.shape[0]

            #! 计算鉴别器的准确率
            discriminator_pred = (discriminator_pred > 0.7).float()
            discriminator_label = discriminator_label.float()
            disc_right += (discriminator_pred == discriminator_label).sum().item()
            disc_all += discriminator_label.shape[0]
            # # 打印TN的数据
            # for i in range(len(discriminator_pred)):
            #     if discriminator_pred[i] ==0 and discriminator_label[i] ==0:
            #         #将print改为logger.info
            #         logger.info('discriminator_pred: {}\tdiscriminator_label: {}'.format(discriminator_pred[i],discriminator_label[i])) 
            #         logger.info('generator_pred:{}'.format(generator_pred[i].int().tolist()))
            #         logger.info('generator_labe:{}'.format(batch_data[3][i][1:].int().tolist()))
            #         logger.info('logits:{}'.format(logits[i].tolist()))
            #         logger.info("\n")
            TP += (discriminator_label * discriminator_pred).sum().item()
            FN += (discriminator_label * (1 - discriminator_pred)).sum().item()
            FP += ((1 - discriminator_label) * discriminator_pred).sum().item()
            TN += ((1 - discriminator_label) * (1 - discriminator_pred)).sum().item()

            
            if corrector_pred is not None:
                #! 计算纠错器对于多有code的准确率，注意这里纠错的是所有的pred_code
                corrector_label = batch_data[3][:, 1:]
                correct_all_pred_right += ((corrector_pred==corrector_label).all(dim=-1).float()).sum().item()
                correct_all_pred_all += code_label.shape[0]

                #! 计算纠错器的准确率，注意这里纠错的仅是所有的错误，不包括鉴别器判定为正确的
                corrector_label = batch_data[3][:, 1:]
                correct_mask = (corrector_label == code_pred).all(dim=-1).float()
                correct_mask = 1 - correct_mask
                corrector_right += ((corrector_pred==corrector_label).all(dim=-1).float() * correct_mask).sum().item()
                corrector_all += correct_mask.sum().item()

                #! 计算最后的准确率，生成器通过鉴别器后，鉴定为错误的code，通过纠错器重新生成code
                final_code_pred = (discriminator_pred==0).float() * corrector_pred + (discriminator_pred==1).float() * code_pred
                final_code_right += (final_code_pred == code_label).all(dim=-1).float().sum().item()
                final_code_all += code_label.shape[0]



    # discriminator_pred和discriminator_label的形状为(batch_size, 1)，计算P,F1,R
    if logger is not None:
        #! 生成器的准确率
        logger.info('code_right: {}\tcode_total: {}\tM-tree codes acc: {}'.format(code_right, code_all, code_right / max(1,code_all)))
        
        #! 鉴别器的准确率
        logger.info('disc_right:{},\tdisc_all:{},\tdiscriminator acc:{}'.format(disc_right,disc_all,disc_right/max(1,disc_all)))
        logger.info('precision:{},\trecall:{},\tF1:{}'.format(TP / max(1,(TP + FP )),TP / max(1,(TP + FN )),2 * TP / max(1,(2 * TP + FP + FN))))
        logger.info('TP:{},TN:{},FN:{},FP:{}'.format(TP,TN,FN,FP))
        
        #!纠错器对于所有code_pred的准确率
        logger.info('correct_all_pred_right: {}\tcorrector_all_pred_total: {}\tcorrector_all_pred acc: {}'.format(correct_all_pred_right, correct_all_pred_all, correct_all_pred_right / max(1,correct_all_pred_all)))


        #! 纠错器的准确率
        logger.info('corrector_right: {}\tcorrector_total: {}\tcorrector acc: {}'.format(corrector_right, corrector_all, corrector_right / max(1,corrector_all)))
        
        #! 最后的准确率
        logger.info('final_code_right: {}\tfinal_code_total: {}\tfinal_code acc: {}'.format(final_code_right, final_code_all, final_code_right / max(1,final_code_all)))


    model.train()

    # return count / total_len
    # return disc_right/disc_all
    return corrector_right/corrector_all
