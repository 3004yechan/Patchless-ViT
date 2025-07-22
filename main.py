import os
import sys
import logging
import pandas as pd
from functools import partial

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision import transforms

# 프로젝트 모듈 임포트
from models import Temp  # 모델 정의 클래스 (models/model.py에서 구현)
from trainer import Trainer  # 학습/검증 루프를 담당하는 트레이너 클래스
from config import get_args  # 명령줄 인수 파싱 함수
from lr_scheduler import get_sch  # 학습률 스케줄러 생성 함수
from utils import seed_everything, handle_unhandled_exception, save_to_json  # 유틸리티 함수들

if __name__ == "__main__":
    # 1. 초기 설정 및 환경 구성
    args = get_args()  # 명령줄 인수 파싱 (학습률, 배치 크기, 에폭 수 등)
    
    # TinyImageNet에 맞게 설정값 변경
    args.num_classes = 200
    args.image_size = 64

    seed_everything(args.seed)  # 재현성을 위한 랜덤 시드 고정
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  # GPU 사용 설정

    # 2. 결과 저장 경로 설정
    if args.continue_train > 0:
        # 이어서 학습하는 경우 기존 폴더 사용
        result_path = args.continue_from_folder
    else:
        # 새로운 학습의 경우 새 폴더 생성 (모델명_폴더번호 형식)
        result_path = os.path.join(args.result_path, args.model +'_'+str(len(os.listdir(args.result_path))))
        os.makedirs(result_path, exist_ok=True)
    
    # 3. 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))  # 로그를 파일로 저장
    logger.info(args)  # 설정값들을 로그에 기록
    save_to_json(vars(args), os.path.join(result_path, 'config.json'))  # 설정을 JSON으로 저장
    sys.excepthook = partial(handle_unhandled_exception,logger=logger)  # 예외 발생 시 로그에 기록

    # 4. 데이터 로딩 (Hugging Face Datasets 사용)
    dataset = load_dataset("zh-plus/tiny-imagenet")

    # 5. 데이터 전처리
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])

    def preprocess_function(examples):
        # image 필드에 transform 적용
        examples['pixel_values'] = [transform(image.convert("RGB")) for image in examples['image']]
        return examples

    processed_datasets = dataset.with_transform(preprocess_function)
    
    train_dataset = processed_datasets['train']
    valid_dataset = processed_datasets['valid']

    # 6. 모델, 손실함수, 옵티마이저, 스케줄러 설정
    model = Temp(args).to(device)  # 모델 생성 후 GPU로 이동
    loss_fn = nn.CrossEntropyLoss()  # 다중 클래스 분류를 위한 교차 엔트로피 손실
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Adam 옵티마이저
    scheduler = get_sch(args.scheduler, optimizer, warmup_epochs=args.warmup_epochs, epochs=args.epochs)  # 학습률 스케줄러

    # 7. 이어서 학습하는 경우 체크포인트 로드
    if args.continue_train_from is not None:
        state = torch.load(args.continue_train_from)  # 저장된 상태 로드
        model.load_state_dict(state['model'])  # 모델 가중치 복원
        optimizer.load_state_dict(state['optimizer'])  # 옵티마이저 상태 복원
        scheduler.load_state_dict(state['scheduler'])  # 스케줄러 상태 복원
        epoch = state['epoch']  # 시작 에폭 설정
    else:
        epoch = 0  # 처음부터 학습하는 경우

    # 8. 데이터로더 생성
    def collate_fn(batch):
        # Trainer가 기대하는 형식에 맞게 키 이름을 'data'와 'label'로 변경
        pixel_values = torch.stack([x['pixel_values'] for x in batch])
        labels = torch.tensor([x['label'] for x in batch])
        return {'data': pixel_values, 'label': labels}

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,  # 학습 시에는 데이터 순서를 섞음
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # 검증 시에는 순서 유지
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    
    # 9. 트레이너 생성 및 학습 시작
    trainer = Trainer(
        train_loader, valid_loader, model, loss_fn, optimizer, scheduler, 
        device, args.patience, args.epochs, result_path, logger, start_epoch=epoch
    )
    trainer.train()  # 학습 루프 실행

    # 10. 테스트 (검증 데이터셋으로 대체) 및 결과 저장
    # TinyImageNet 'test' split에는 레이블이 없으므로, 'valid' split으로 테스트를 대신합니다.
    # submission 파일 생성 로직은 실제 테스트셋에 맞게 조정 필요
    logger.info("="*20 + " Test Start " + "="*20)
    
    # test_loader를 valid_loader로 대체하여 예측 수행
    predictions = trainer.test(valid_loader) # trainer.test가 softmax 확률을 반환한다고 가정

    # 예측 결과를 데이터프레임에 저장 (예시 로직)
    # 실제 'id'가 없으므로 인덱스를 사용
    image_ids = list(range(len(valid_dataset)))
    
    # 제출 파일 형식에 맞게 데이터프레임 생성
    submission_df = pd.DataFrame()
    submission_df['id'] = image_ids
    
    # 예측 확률을 컬럼으로 추가
    pred_cols = [f'class_{i}' for i in range(args.num_classes)]
    pred_df = pd.DataFrame(predictions, columns=pred_cols)
    
    submission_df = pd.concat([submission_df, pred_df], axis=1)
    
    submission_df.to_csv(os.path.join(result_path, 'submission.csv'), index=False)
    logger.info("Test finished. Submission file saved to {}".format(os.path.join(result_path, 'submission.csv'))) 