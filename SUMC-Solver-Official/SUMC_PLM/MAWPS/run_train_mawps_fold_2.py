import argparse
import json
import os


parser = argparse.ArgumentParser()
#! 选择设备，保存模型的name，训练or测试
parser.add_argument("--gpu_device", default="0", type=str)

parser.add_argument("--model_name", default="corrector2",type=str)
parser.add_argument("--mode", default="train", type=str)

#! 加载训练好的模型
parser.add_argument("--fc_path", default="output/model_save_name2/best_model/fc_weight.bin", type=str)
parser.add_argument("--pretrain_model_path",default="output/model_save_name2/best_model",type=str)# bert-base-uncased
parser.add_argument("--discriminator_path", default="output/discriminator2/best_model", type=str) #output/discriminator2/best_model
parser.add_argument("--corrector_path", default=None, type=str) # output/corrector1/best_model 

parser.add_argument("--num_labels", default=28, type=int)

parser.add_argument("--train_type", default='one-by-one-random', type=str,
                    help='one-by-one-in-same-batch or one-by-one-random or together')

parser.add_argument('--re_process_train_data', default=False, type=bool)

parser.add_argument("--deal_data_imbalance", default=0, type=int)

parser.add_argument("--multi_fc", default=True, type=bool)
parser.add_argument("--fc_hidden_size", default=2048, type=int)

parser.add_argument("--use_cls", default=True, type=bool)
parser.add_argument("--train_max_len", default=100, type=int)
parser.add_argument("--test_dev_max_len", default=100, type=int)
parser.add_argument("--use_new_token_type_id", default=True, type=bool)
parser.add_argument("--train_loss", default='MSE', type=str, help='MSE or L1 or Huber')

parser.add_argument("--use_multi_gpu", default=False, type=bool)
parser.add_argument("--batch_size", default=20, type=int)
parser.add_argument("--num_epochs", default=120, type=int)
parser.add_argument("--log_steps_per_epoch", default=10, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--warmup", default=0.1, type=float)



parser.add_argument("--train_data_path",
                    default="./mawps/Fold_2/train_mawps_new_mwpss_fold_2.json",
                    type=str)

parser.add_argument("--dev_data_path",
                    default="./mawps/Fold_2/test_mawps_new_mwpss_fold_2.json",
                    type=str)

parser.add_argument("--label2id_path", default='../../dataset/codes_mawps.json',
                    type=str)


parser.add_argument("--output_dir", default="./output/", type=str)


args = parser.parse_args()
args_dict = args.__dict__
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
from src.Train import train

if __name__ == "__main__":

    out_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'setting_parameters.json'), 'w', encoding='utf-8')as f:
        f.write(json.dumps(args_dict, ensure_ascii=False, indent=2))
    train(args)
