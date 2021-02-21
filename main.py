import argparse
from Test.test_PretrainBert import test_PretrainBert
from Test.test_Bert import test_Bert
from Test.test_DKT import test_DKT

from Dataset.data_helper import setup_seed, setup_pandas

parser = argparse.ArgumentParser(description='Hello')

# model
parser.add_argument('--model', type=str, default='Bert')
parser.add_argument('--dropout',type=float,default=0.1)
parser.add_argument('--num_heads',type=int,default=12)
parser.add_argument('--num_layers',type=int,default=12)
parser.add_argument('--hidden_dim',type=int,default=768 * 4)
parser.add_argument('--embedding_dim',type=int,default=768)
parser.add_argument('--input_name',type=str,default='[\'st+ct_sequence\']')
parser.add_argument('--label_name',type=str,default='last_correctness')
parser.add_argument('--query_name',type=str,default='last_skill_id')
parser.add_argument('--use_combine_linear',action='store_true', default=False)

# dataset
parser.add_argument('--data_root_dir', type=str, default='data')
parser.add_argument('--dataset', type=str, default='Assistment09')
parser.add_argument('--trunk_size',type=int,default=35)

# logs
parser.add_argument('--tensorboard_log_dir', type=str, default='tensorboard_logs')
parser.add_argument('--models_checkpoints_dir',type=str,default='models_checkpoints')
parser.add_argument('--log_name',type=str,default='test')

# pretrain
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--use_pretrained_model', action='store_true', default=False)
parser.add_argument('--pretrained_model_name',type=str,default='PretrainedBert.pt')

# train
parser.add_argument('--lr', type=float, default=2e-5)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--cuda', type=str, default='cuda:0')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--pin_memory', action='store_true', default=False)
parser.add_argument('--patience', type=int, default=100)

# optimizer
parser.add_argument('--optimizer', type=str, default='AdamW')
parser.add_argument('--weight_decay', type=float, default=1e-2)
parser.add_argument('--step_size', type=int, default=1)
parser.add_argument('--gamma',type=float, default=1.0)

args = parser.parse_args()

if __name__ == '__main__':
    setup_seed(41)
    setup_pandas()

    if args.model == 'PretrainBert':
        test_PretrainBert(args)
    elif args.model == 'DKT':
        test_DKT(args)
    elif args.model == 'Bert':
        test_Bert(args)
