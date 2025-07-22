import os
import sys
import logging
import pandas as pd
from functools import partial
from sklearn.model_selection import train_test_split

import torch
from torch import optim, nn
from torch.utils.data import DataLoader

# 프로젝트 모듈 임포트
from models import Temp  # 모델 정의 클래스 (models/model.py에서 구현)
from data import DataSet  # 커스텀 데이터셋 클래스 (data.py에서 구현)
from trainer import Trainer  # 학습/검증 루프를 담당하는 트레이너 클래스
from config import get_args  # 명령줄 인수 파싱 함수
from lr_scheduler import get_sch  # 학습률 스케줄러 생성 함수
from utils import seed_everything, handle_unhandled_exception, save_to_json  # 유틸리티 함수들

if __name__ == "__main__":
    # 1. 초기 설정 및 환경 구성
    args = get_args()  # 명령줄 인수 파싱 (학습률, 배치 크기, 에폭 수 등)
    seed_everything(args.seed)  # 재현성을 위한 랜덤 시드 고정
    device = torch.device('cuda:0')  # GPU 사용 설정

    # 2. 결과 저장 경로 설정
    if args.continue_train > 0:
        # 이어서 학습하는 경우 기존 폴더 사용
        result_path = args.continue_from_folder
    else:
        # 새로운 학습의 경우 새 폴더 생성 (모델명_폴더번호 형식)
        result_path = os.path.join(args.result_path, args.model +'_'+str(len(os.listdir(args.result_path))))
        os.makedirs(result_path)
    
    # 3. 로깅 설정
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.addHandler(logging.FileHandler(os.path.join(result_path, 'log.log')))  # 로그를 파일로 저장
    logger.info(args)  # 설정값들을 로그에 기록
    save_to_json(vars(args), os.path.join(result_path, 'config.json'))  # 설정을 JSON으로 저장
    sys.excepthook = partial(handle_unhandled_exception,logger=logger)  # 예외 발생 시 로그에 기록

    # 4. 데이터 로딩 및 경로 설정
    train_data = pd.read_csv(args.train)  # 학습 데이터 CSV 파일 읽기
    train_data['path'] = train_data['path'].apply(lambda x: os.path.join(args.path, x))  # 상대 경로를 절대 경로로 변환
    test_data = pd.read_csv(args.test)  # 테스트 데이터 CSV 파일 읽기
    test_data['path'] = test_data['path'].apply(lambda x: os.path.join(args.path, x))  # 테스트 데이터 경로도 절대 경로로 변환

    # 5. 모델 입출력 크기 설정 (현재는 None으로 초기화)
    input_size = None  # 입력 데이터의 차원 (실제 구현에서는 데이터에 따라 설정)
    output_size = None  # 출력 클래스 수 (분류 문제의 경우 클래스 개수)

    # 6. 예측 결과 및 앙상블을 위한 데이터프레임 준비
    prediction = pd.read_csv(args.submission)  # 제출용 템플릿 파일 읽기
    output_index = [f'{i}' for i in range(0, output_size)]  # 출력 클래스별 컬럼명 생성
    stackking_input = pd.DataFrame(columns = output_index, index=range(len(train_data)))  # 스태킹 앙상블용 OOF 예측 저장 공간
  
    # 7. 데이터 분할 (학습용/검증용)
    train_index, valid_index = train_test_split(
        range(len(train_data['path'])), 
        test_size=args.test_size, 
        shuffle=True, 
        random_state=args.seed, 
        stratify=train_data['label']  # 계층화 샘플링으로 라벨 분포 유지
    )
    seed_everything(args.seed)  # 데이터 분할 후 시드 재고정

    # 8. 분할된 데이터로 데이터셋 생성
    kfold_train_data = train_data.iloc[train_index]  # 학습용 데이터 추출
    kfold_valid_data = train_data.iloc[valid_index]  # 검증용 데이터 추출

    # 9. PyTorch 데이터셋 객체 생성
    train_dataset = DataSet(file_list=kfold_train_data['path'].values, label=kfold_train_data['label'].values)
    valid_dataset = DataSet(file_list=kfold_valid_data['path'].values, label=kfold_valid_data['label'].values)

    # 10. 모델, 손실함수, 옵티마이저, 스케줄러 설정
    model = Temp(args).to(device)  # 모델 생성 후 GPU로 이동
    loss_fn = nn.CrossEntropyLoss()  # 다중 클래스 분류를 위한 교차 엔트로피 손실
    optimizer = optim.Adam(model.parameters(), lr=args.lr)  # Adam 옵티마이저
    scheduler = get_sch(args.scheduler, optimizer, warmup_epochs=args.warmup_epochs, epochs=args.epochs)  # 학습률 스케줄러

    # 11. 이어서 학습하는 경우 체크포인트 로드
    if args.continue_train_from is not None:
        state = torch.load(args.continue_train_from)  # 저장된 상태 로드
        model.load_state_dict(state['model'])  # 모델 가중치 복원
        optimizer.load_state_dict(state['optimizer'])  # 옵티마이저 상태 복원
        scheduler.load_state_dict(state['scheduler'])  # 스케줄러 상태 복원
        epoch = state['epoch']  # 시작 에폭 설정
    else:
        epoch = 0  # 처음부터 학습하는 경우

    # 12. 데이터로더 생성
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,  # 학습 시에는 데이터 순서를 섞음
        num_workers=args.num_workers,  # 멀티프로세싱 워커 수
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # 검증 시에는 순서 유지
        num_workers=args.num_workers,
    )
    
    # 13. 트레이너 생성 및 학습 시작
    trainer = Trainer(
        train_loader, valid_loader, model, loss_fn, optimizer, scheduler, 
        device, args.patience, args.epochs, result_path, logger, start_epoch=epoch
    )
    trainer.train()  # 학습 루프 실행

    # 14. 테스트 데이터에 대한 예측 수행
    test_dataset = DataSet(file_list=test_data['path'].values, label=test_data['label'].values)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,  # 테스트 시에는 순서 유지 필수
        num_workers=args.num_workers
    )

    # 15. 최종 예측 결과 저장
    prediction[output_index] += trainer.test(test_loader)  # 테스트 예측 수행 (softmax 적용된 확률값)
    prediction.to_csv(os.path.join(result_path, 'sum.csv'), index=False)  # 예측 결과를 CSV로 저장 