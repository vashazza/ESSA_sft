# Qwen3-4B-Thinking Fine-tuning Pipeline

## 📁 프로젝트 구조

```
├── README.md                    # 프로젝트 설명서
├── requirements.txt             # 패키지
├── 
├── src/                        # 소스 코드
│   ├── train_qwen_thinking.py  # 메인 훈련 스크립트
│   ├── data_preprocessing.py   # 데이터 전처리
│   ├── thinking_data_collator.py # Thinking 모델 전용 콜레이터
│   ├── model_utils.py          # 유틸리티
│   └── inference.py            
├── 
├── config/                     # 설정 파일
│   └── training_config.yaml
├── 
└── data/                       # 훈련 데이터
    ├── general_only_data/      # 도메인 태스크 구분 없는 데이터
    ├── sft_data/               # 최종 데이터
    └── no_spec_data/           # spec 없는 데이터
```

