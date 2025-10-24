#!/usr/bin/env python3
"""
Qwen3-4B-Thinking-2507 Fine-tuning Script
YAML 설정 파일을 사용한 유연한 fine-tuning 스크립트
"""

import os
import sys
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import TrainingArguments, EarlyStoppingCallback, set_seed
from trl import SFTTrainer

# 로컬 모듈 임포트
from model_utils import (
    load_model_and_tokenizer, 
    load_datasets, 
    format_chat_template,
    setup_wandb,
    save_model_and_config
)
from thinking_data_collator import create_data_collator

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class QwenThinkingTrainer:
    """Qwen3 Thinking 모델 훈련 관리자"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.setup_environment()
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """YAML 설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # timestamp 동적 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if 'training' in config and 'output_dir' in config['training']:
            config['training']['output_dir'] = config['training']['output_dir'].format(
                timestamp=timestamp
            )
            
        return config
    
    def setup_environment(self):
        """환경 설정"""
        # 시드 설정
        seed = self.config.get('training', {}).get('seed', 42)
        set_seed(seed)
        
        # 로깅 레벨 설정
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        logging.getLogger().setLevel(log_level)
        
        # 멀티 GPU 환경 설정
        hardware_config = self.config.get('hardware', {})
        if hardware_config.get('multi_gpu', False):
            # DDP 환경 변수 설정
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                num_gpus = hardware_config.get('num_gpus', 2)
                os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(num_gpus)))
            
            # NCCL 백엔드 설정
            if hardware_config.get('ddp_backend') == 'nccl':
                os.environ['NCCL_DEBUG'] = 'INFO'
                os.environ['NCCL_SOCKET_IFNAME'] = 'lo'  # 로컬 인터페이스 사용
            
            logger.info(f"Multi-GPU setup: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
            logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        
        # Wandb 설정
        setup_wandb(self.config)
        
        logger.info("Environment setup completed")
    
    def create_training_arguments(self, train_dataset_size: int = None) -> TrainingArguments:
        """TrainingArguments 생성"""
        training_config = self.config['training']
        data_config = self.config['data']
        hardware_config = self.config.get('hardware', {})
        
        # 멀티 GPU 설정에 따른 배치 크기 조정
        if hardware_config.get('multi_gpu', False):
            # DDP에서는 전체 배치 크기가 GPU 수만큼 곱해지므로 조정
            num_gpus = hardware_config.get('num_gpus', 2)
            per_device_batch_size = max(1, training_config.get('per_device_train_batch_size', 1))
            logger.info(f"Multi-GPU: per_device_batch_size={per_device_batch_size}, total_effective_batch_size={per_device_batch_size * num_gpus}")
        else:
            num_gpus = 1
            per_device_batch_size = training_config.get('per_device_train_batch_size', 1)
        
        # 0.5 에폭마다 체크포인트를 위한 스텝 계산
        eval_steps = None
        save_steps = None
        if train_dataset_size is not None:
            gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 16)
            effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_gpus
            steps_per_epoch = train_dataset_size // effective_batch_size
            if train_dataset_size % effective_batch_size != 0:
                steps_per_epoch += 1
            
            # 0.5 에폭마다 = steps_per_epoch // 2
            half_epoch_steps = max(1, steps_per_epoch // 2)
            eval_steps = training_config.get('eval_steps', half_epoch_steps)
            save_steps = training_config.get('save_steps', half_epoch_steps)
            
            logger.info(f"Dataset size: {train_dataset_size}")
            logger.info(f"Steps per epoch: {steps_per_epoch}")
            logger.info(f"Checkpoint every {half_epoch_steps} steps (0.5 epochs)")
        
        return TrainingArguments(
            # 기본 설정
            output_dir=training_config['output_dir'],
            overwrite_output_dir=training_config.get('overwrite_output_dir', False),
            do_train=training_config.get('do_train', True),
            do_eval=training_config.get('do_eval', True),
            
            # 배치 및 그래디언트 설정
            per_device_train_batch_size=training_config.get('per_device_train_batch_size', 1),
            per_device_eval_batch_size=training_config.get('per_device_eval_batch_size', 1),
            gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 16),
            dataloader_num_workers=training_config.get('dataloader_num_workers', 0),
            
            # 옵티마이저 설정
            learning_rate=training_config.get('learning_rate', 5e-5),
            weight_decay=training_config.get('weight_decay', 0.0),
            adam_beta1=training_config.get('adam_beta1', 0.9),
            adam_beta2=training_config.get('adam_beta2', 0.999),
            adam_epsilon=training_config.get('adam_epsilon', 1e-8),
            max_grad_norm=training_config.get('max_grad_norm', 1.0),
            
            # 스케줄러 설정
            lr_scheduler_type=training_config.get('lr_scheduler_type', 'cosine'),
            warmup_ratio=training_config.get('warmup_ratio', 0.03),
            
            # 에폭 및 스텝 설정
            num_train_epochs=training_config.get('num_train_epochs', 3.0),
            max_steps=training_config.get('max_steps', -1),
            
            # 로깅 및 평가 설정
            logging_steps=training_config.get('logging_steps', 10),
            eval_strategy=training_config.get('eval_strategy', 'steps'),
            eval_steps=eval_steps,  # 0.5 에폭마다 평가
            save_strategy=training_config.get('save_strategy', 'steps'),
            save_steps=save_steps,  # 0.5 에폭마다 저장
            save_total_limit=training_config.get('save_total_limit', 6),  # 더 많은 체크포인트 보관
            load_best_model_at_end=training_config.get('load_best_model_at_end', True),
            metric_for_best_model=training_config.get('metric_for_best_model', 'eval_loss'),
            greater_is_better=training_config.get('greater_is_better', False),
            
            # 정밀도 및 메모리 최적화
            fp16=training_config.get('fp16', False),
            bf16=training_config.get('bf16', True),
            gradient_checkpointing=training_config.get('gradient_checkpointing', False),
            ddp_find_unused_parameters=training_config.get('ddp_find_unused_parameters', False),
            remove_unused_columns=training_config.get('remove_unused_columns', True),
            dataloader_pin_memory=training_config.get('dataloader_pin_memory', False),
            
            # 실험 추적
            report_to=training_config.get('report_to', 'wandb'),
            run_name=training_config.get('run_name', 'qwen3-4b-thinking-sft'),
            seed=training_config.get('seed', 42),
        )
    
    def create_trainer(self, model, tokenizer, train_dataset, val_dataset) -> SFTTrainer:
        """SFTTrainer 생성"""
        training_args = self.create_training_arguments(len(train_dataset))
        data_config = self.config['data']
        training_config = self.config['training']
        
        # 데이터 콜레이터 생성
        data_collator = create_data_collator(tokenizer, self.config)
        
        # 콜백 설정
        callbacks = []
        if training_config.get('early_stopping_patience'):
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=training_config['early_stopping_patience']
                )
            )
        
        # SFTTrainer 생성 (TRL 0.23.1 호환)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            
            # 콜백
            callbacks=callbacks if callbacks else None,
        )
        
        return trainer
    
    def train(self):
        """전체 훈련 프로세스 실행"""
        logger.info("🚀 Starting Qwen3-4B-Thinking fine-tuning...")
        
        # 1. 모델과 토크나이저 로드
        logger.info("📦 Loading model and tokenizer...")
        model, tokenizer = load_model_and_tokenizer(self.config)
        
        # 2. 데이터셋 로드
        logger.info("📊 Loading datasets...")
        train_dataset, val_dataset = load_datasets(self.config)
        
        # 3. 트레이너 생성
        logger.info("🔧 Creating trainer...")
        trainer = self.create_trainer(model, tokenizer, train_dataset, val_dataset)
        
        # 4. 훈련 시작
        logger.info("🎯 Starting training...")
        trainer.train()
        
        # 5. 모델 저장
        logger.info("💾 Saving model...")
        save_model_and_config(trainer, self.config)
        
        logger.info("✅ Training completed successfully!")
        
        return trainer


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Qwen3-4B-Thinking Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 설정으로 훈련
  python train_qwen_thinking.py
  
  # 커스텀 설정 파일 사용
  python train_qwen_thinking.py --config my_config.yaml
  
  # 체크포인트에서 재시작
  python train_qwen_thinking.py --resume_from_checkpoint ./checkpoints/checkpoint-1000
        """
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="training_config.yaml",
        help="Path to training configuration YAML file"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    
    args = parser.parse_args()
    
    # 설정 파일 존재 확인
    if not os.path.exists(args.config):
        logger.error(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    try:
        # 훈련 시작
        trainer_manager = QwenThinkingTrainer(args.config)
        trainer = trainer_manager.train()
        
        logger.info("🎉 All done! Happy fine-tuning!")
        
    except KeyboardInterrupt:
        logger.info("⚠️ Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()