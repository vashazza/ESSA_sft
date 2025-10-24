#!/usr/bin/env python3
"""
모델 및 토크나이저 로딩 유틸리티
"""

import os
import torch
import logging
from typing import Dict, Any, Tuple
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """모델과 토크나이저 로드"""
    model_config = config['model']
    hardware_config = config.get('hardware', {})
    
    logger.info(f"Loading model: {model_config['name_or_path']}")
    
    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        model_config['name_or_path'],
        cache_dir=model_config.get('cache_dir'),
        use_fast=model_config.get('use_fast_tokenizer', True),
        trust_remote_code=model_config.get('trust_remote_code', True),
        revision=model_config.get('model_revision', 'main')
    )
    
    # 패딩 토큰 설정
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
        
    # Flash Attention 설정
    model_kwargs = {
        'cache_dir': model_config.get('cache_dir'),
        'torch_dtype': getattr(torch, model_config.get('dtype', 'bfloat16')),
        'trust_remote_code': model_config.get('trust_remote_code', True),
        'revision': model_config.get('model_revision', 'main'),
        'low_cpu_mem_usage': hardware_config.get('low_cpu_mem_usage', True)
    }
    
    # Flash Attention 2 설정
    if hardware_config.get('use_flash_attention', True):
        model_kwargs['attn_implementation'] = hardware_config.get('attn_implementation', 'flash_attention_2')
        logger.info(f"Using Flash Attention: {model_kwargs['attn_implementation']}")
    
    # 멀티 GPU 설정
    if hardware_config.get('multi_gpu', False):
        # 멀티 GPU 환경에서는 device_map을 None으로 설정 (DDP가 처리)
        model_kwargs['device_map'] = None
        logger.info(f"Multi-GPU mode enabled with {hardware_config.get('num_gpus', 2)} GPUs")
    else:
        model_kwargs['device_map'] = hardware_config.get('device_map', 'auto')
        model_kwargs['max_memory'] = hardware_config.get('max_memory')
    
    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        model_config['name_or_path'],
        **model_kwargs
    )
    
    # LoRA 설정 적용
    if model_config.get('use_lora', False):
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=model_config.get('lora_rank', 16),
            lora_alpha=model_config.get('lora_alpha', 32),
            lora_dropout=model_config.get('lora_dropout', 0.05),
            target_modules=model_config.get('lora_target_modules', [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            bias="none"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        logger.info("LoRA configuration applied")
        
    return model, tokenizer


def load_datasets(config: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    """데이터셋 로드 및 전처리"""
    data_config = config['data']
    
    logger.info("Loading datasets...")
    
    # 데이터셋 로드
    train_dataset = load_dataset(
        'json', 
        data_files=data_config['train_dataset_path'],
        split='train'
    )
    
    val_dataset = None
    if data_config.get('validation_dataset_path'):
        val_dataset = load_dataset(
            'json',
            data_files=data_config['validation_dataset_path'], 
            split='train'
        )
    
    # 데이터 필터링
    max_samples = data_config.get('max_samples_per_dataset')
    if max_samples:
        train_dataset = train_dataset.select(range(min(len(train_dataset), max_samples)))
        if val_dataset:
            val_dataset = val_dataset.select(range(min(len(val_dataset), max_samples)))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
    return train_dataset, val_dataset


def format_chat_template(example: Dict[str, Any], tokenizer: AutoTokenizer, config: Dict[str, Any]) -> Dict[str, str]:
    """채팅 템플릿 적용 (Thinking 지원)"""
    enable_thinking = config['model'].get('enable_thinking', True)
    
    # 메시지 형식으로 변환
    messages = []
    if 'messages' in example:
        messages = example['messages']
    elif 'conversations' in example:
        for turn in example['conversations']:
            messages.append({
                "role": turn.get('from', 'user'), 
                "content": turn.get('value', '')
            })
    else:
        # 기본 형식 처리
        if 'instruction' in example and 'output' in example:
            messages = [
                {"role": "user", "content": example['instruction']},
                {"role": "assistant", "content": example['output']}
            ]
    
    # 채팅 템플릿 적용
    try:
        if hasattr(tokenizer, 'apply_chat_template'):
            # Thinking 모델의 경우 enable_thinking 파라미터 사용
            if enable_thinking:
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                    enable_thinking=True  # Thinking 활성화
                )
            else:
                formatted_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False
                )
        else:
            # 기본 포맷팅
            formatted_text = ""
            for msg in messages:
                if msg['role'] == 'user':
                    formatted_text += f"User: {msg['content']}\n"
                elif msg['role'] == 'assistant':
                    formatted_text += f"Assistant: {msg['content']}\n"
                    
    except Exception as e:
        logger.warning(f"Chat template application failed: {e}")
        # 폴백 포맷팅
        formatted_text = ""
        for msg in messages:
            formatted_text += f"{msg['role'].title()}: {msg['content']}\n"
    
    return {"text": formatted_text}


def setup_wandb(config: Dict[str, Any]) -> None:
    """Wandb 설정"""
    wandb_config = config.get('wandb', {})
    
    if not wandb_config.get('project'):
        logger.info("Wandb project not specified, skipping wandb setup")
        return
    
    # 환경변수 설정
    if wandb_config.get('project'):
        os.environ.setdefault('WANDB_PROJECT', wandb_config['project'])
    
    if wandb_config.get('entity'):
        os.environ.setdefault('WANDB_ENTITY', wandb_config['entity'])
    
    # Wandb 초기화
    try:
        import wandb
        wandb.init(
            project=wandb_config.get('project'),
            entity=wandb_config.get('entity'),
            name=config.get('training', {}).get('run_name', 'qwen3-4b-thinking-sft'),
            tags=wandb_config.get('tags', []),
            config=config,
            reinit=True
        )
        logger.info(f"Wandb initialized: project={wandb_config.get('project')}")
    except ImportError:
        logger.warning("Wandb not available, skipping initialization")
    except Exception as e:
        logger.warning(f"Failed to initialize wandb: {e}")


def save_model_and_config(trainer, config: Dict[str, Any]) -> None:
    """모델과 설정 저장"""
    logger.info("Saving final model...")
    
    # 모델 저장
    trainer.save_model()
    
    # 토크나이저 저장
    trainer.tokenizer.save_pretrained(trainer.args.output_dir)
    
    # 설정 파일도 함께 저장
    import yaml
    config_save_path = Path(trainer.args.output_dir) / "training_config.yaml"
    with open(config_save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    logger.info(f"Model and config saved to: {trainer.args.output_dir}")
