import argparse

def args_for_data(parser):
    parser.add_argument('--train', type=str, default='../data/train.csv')
    parser.add_argument('--test', type=str, default='../data/test.csv')
    parser.add_argument('--submission', type=str, default='../data/sample_submission.csv')
    parser.add_argument('--path', type=str, default='../data')
    parser.add_argument('--result_path', type=str, default='./result')
    
def args_for_train(parser):
    parser.add_argument('--cv_k', type=int, default=10, help='k-fold stratified cross validation')
    parser.add_argument('--test_size', type=int, default=0.3, help='test size for stratified train-test split, (main.py)')
    parser.add_argument('--num_workers', type=int, default=4, help='num_workers')
    parser.add_argument('--batch_size', type=int, default=None, help='batch_size')
    parser.add_argument('--epochs', type=int, default=1000, help='max epochs')
    parser.add_argument('--patience', type=int, default=15, help='patience for early stopping')    
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate for the optimizer')
    parser.add_argument('--scheduler', type=str, default='None')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of warmup epoch of lr scheduler')

    # ViT 모델을 위한 하이퍼파라미터 추가
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes for the model')
    parser.add_argument('--image_size', type=int, default=224, help='input image size for the model')

    parser.add_argument('--continue_train', type=int, default=-1, help='continue training from fold x') 
    parser.add_argument('--continue_train_from', type=str, default=None, help='continue training from last model, (main.py)') 
    parser.add_argument('--continue_from_folder', type=str, help='continue training from args.continue_from')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int)
    # 모델 선택 인자 수정
    parser.add_argument('--model', default='vit_b16', type=str,
                        help='model type to use (vit_b16 or sparse_vit_b16)',
                        choices=['vit_b16', 'sparse_vit_b16'])

    args_for_data(parser)
    args_for_train(parser)
    
    # 더 이상 필요 없는 args_for_model 호출 제거
    # _args, _ = parser.parse_known_args()
    # args_for_model(parser, _args.model)

    args = parser.parse_args()
    return args
