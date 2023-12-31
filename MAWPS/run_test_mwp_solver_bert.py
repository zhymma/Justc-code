import argparse

from src.Test_new import test_for_mwp_BERT_slover

parser = argparse.ArgumentParser()
parser.add_argument("--log_file", default="output.log", type=str)



parser.add_argument("--batch_size", default=1, type=int)
parser.add_argument("--multi_fc", default=True, type=bool)

parser.add_argument("--test_dev_max_len", default=100, type=int)
parser.add_argument("--use_cls", default=True, type=bool)

parser.add_argument("--test_data_path",

                    default='./mawps/Fold_0/test_mawps_new_mwpss_fold_0.json',
                    type=str)
parser.add_argument("--num_labels", default=28, type=int)

parser.add_argument("--pretrain_model_path_for_test",
                    default="output/test/best_model",
                    type=str)

parser.add_argument("--label2id_path", default='mawps/codes_mawps.json',
                    type=str)

parser.add_argument("--gpu_device", default="3", type=str)

parser.add_argument("--use_new_token_type_id", default=True, type=bool)
parser.add_argument("--use_multi_gpu", default=False, type=bool)
parser.add_argument("--fc_hidden_size", default=2048, type=int)
parser.add_argument("--train_loss", default='MSE', type=str, help='MSE or L1 or Huber')

if __name__ == "__main__":
    args = parser.parse_args()
    print(args.test_data_path)
    print(args.pretrain_model_path_for_test)

    test_for_mwp_BERT_slover(args,"false.json")
