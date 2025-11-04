#!/usr/bin/env python3
"""
모델 응답 생성 시스템
베이스 모델과 파인튜닝 모델의 응답을 생성하여 저장
"""

import os
import json
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import pandas as pd
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)




class ModelInference:
    """베이스 모델과 파인튜닝 모델 응답 생성"""
    
    def __init__(self, config_path: str, base_model_path: str, finetuned_model_path: str):
        self.config = self.load_config(config_path)
        self.base_model_path = base_model_path
        self.finetuned_model_path = finetuned_model_path
        
        # 결과 저장 경로 설정
        eval_config = self.config.get('evaluation', {})
        self.save_path = Path(eval_config.get('eval_save_path', './inference_results'))
        self.save_path.mkdir(parents=True, exist_ok=True)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_models_and_tokenizers(self):
        """베이스 모델과 파인튜닝 모델을 각각 다른 GPU에 로드"""
        logger.info(f"Loading base model from: {self.base_model_path}")
        logger.info(f"Loading finetuned model from: {self.finetuned_model_path}")
        
        # 베이스 모델 토크나이저 (thinking 기능 지원)
        base_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,
            trust_remote_code=True
        )
        
        # 베이스 모델 (GPU 0)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},  # GPU 0에 고정
            trust_remote_code=True
        )
        
        # 파인튜닝 모델 (GPU 1) - 토크나이저는 베이스 모델 것을 사용
        # LoRA 파인튜닝의 경우 토크나이저는 변경되지 않으므로 베이스 모델 토크나이저 사용
        finetuned_tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_path,  # 베이스 모델 경로에서 토크나이저 로드
            trust_remote_code=True
        )
        finetuned_model = AutoModelForCausalLM.from_pretrained(
            self.finetuned_model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 1},  # GPU 1에 고정
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정
        if base_tokenizer.pad_token is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        if finetuned_tokenizer.pad_token is None:
            finetuned_tokenizer.pad_token = finetuned_tokenizer.eos_token
            
        return (base_model, base_tokenizer), (finetuned_model, finetuned_tokenizer)
    
    def load_test_dataset(self) -> List[Dict[str, str]]:
        """False Reject Test 데이터셋 로드"""
        eval_config = self.config.get('evaluation', {})
        max_samples = eval_config.get('max_eval_samples', None)  # 기본값을 None으로 설정
        
        # False Reject Test CSV 파일
        csv_file_path = "safety_layer_data/false_reject_test_strat3_testonly.csv"
        
        questions_answers = []
        
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"False Reject Test file not found: {csv_file_path}")
            
        logger.info(f"Loading data from: {csv_file_path}")
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_file_path)
            
            for idx, row in df.iterrows():
                questions_answers.append({
                    "question": row['prompt'],
                    "answer": "",  # 평가용이므로 정답이 없음
                    "category": row.get('category_text', 'unknown'),
                    "category_id": row.get('category', 'unknown'),
                    "reference": row.get('reference', ''),
                    "label": 'false_reject_test',
                    "id": f'false-reject-{idx}',
                    "source": "false_reject_test_strat3"
                })
                
        except ImportError:
            logger.error("pandas required for CSV files. Install with: pip install pandas")
            raise
        except Exception as e:
            logger.error(f"Failed to load CSV file {csv_file_path}: {e}")
            raise
        
        # 샘플 수 제한 (max_samples가 지정된 경우에만)
        if max_samples and len(questions_answers) > max_samples:
            questions_answers = questions_answers[:max_samples]
            logger.info(f"Limited to {max_samples} samples")
        
        logger.info(f"Loaded {len(questions_answers)} False Reject Test samples")
        
        # 카테고리별 분포 출력
        category_counts = {}
        for qa in questions_answers:
            category = qa.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        
        logger.info(f"Sample distribution by category:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} samples")
        
        return questions_answers
    
    
    def generate_response(self, model, tokenizer, question: str) -> str:
        """모델로 응답 생성"""
        inference_config = self.config.get('inference', {})
        enable_thinking = inference_config.get('enable_thinking', True)
        
        # 메시지 형식으로 변환
        messages = [{"role": "user", "content": question}]
        
        # 채팅 템플릿 적용
        try:
            if enable_thinking and hasattr(tokenizer, 'apply_chat_template'):
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
        except:
            # 폴백
            prompt = f"User: {question}\nAssistant:"
        
        # 토크나이징
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=inference_config.get('max_new_tokens', 2048),
                temperature=inference_config.get('temperature', 0.6),
                top_p=inference_config.get('top_p', 0.95),
                top_k=inference_config.get('top_k', 20),
                do_sample=inference_config.get('do_sample', True),
                repetition_penalty=inference_config.get('repetition_penalty', 1.05),
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # 디코딩
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def generate_response_concurrent(self, base_model, base_tokenizer, finetuned_model, finetuned_tokenizer, question: str):
        """두 모델의 응답을 동시에 생성"""
        def generate_base():
            return self.generate_response(base_model, base_tokenizer, question)
        
        def generate_finetuned():
            return self.generate_response(finetuned_model, finetuned_tokenizer, question)
        
        # ThreadPoolExecutor를 사용하여 병렬 실행
        with ThreadPoolExecutor(max_workers=2) as executor:
            # 두 작업을 동시에 시작
            base_future = executor.submit(generate_base)
            finetuned_future = executor.submit(generate_finetuned)
            
            # 결과 대기
            base_response = base_future.result()
            finetuned_response = finetuned_future.result()
            
            return base_response, finetuned_response
    
    def generate_responses(self):
        """베이스 모델과 파인튜닝 모델 응답 생성 - 실시간 저장"""
        logger.info("Starting model response generation with streaming save...")
        
        # 모델들 로드
        (base_model, base_tokenizer), (finetuned_model, finetuned_tokenizer) = self.load_models_and_tokenizers()
        
        # 테스트 데이터 로드
        test_data = self.load_test_dataset()
        
        # 실시간 저장을 위한 파일 경로 설정
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        responses_path = self.save_path / f"model_responses_{timestamp}.jsonl"
        final_json_path = self.save_path / f"model_responses_{timestamp}.json"
        
        logger.info(f"Real-time saving to: {responses_path}")
        logger.info(f"Final JSON will be saved to: {final_json_path}")
        
        # 결과 수집을 위한 리스트 (최종 JSON 저장용)
        all_responses = []
        
        # 각 쿼리에 대해 동시 처리하고 즉시 저장
        for i, qa in enumerate(tqdm(test_data, desc="Processing queries")):
            logger.info(f"Processing query {i+1}/{len(test_data)}: {qa['question'][:50]}...")
            
            # 두 모델 응답을 동시에 생성
            logger.info("  Generating responses from both models simultaneously...")
            start_time = datetime.now()
            base_response, finetuned_response = self.generate_response_concurrent(
                base_model, base_tokenizer, 
                finetuned_model, finetuned_tokenizer, 
                qa['question']
            )
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            logger.info(f"  Both responses generated in {generation_time:.2f} seconds")
            
            # 응답 데이터 즉시 저장 (JSONL 형식)
            response_data = {
                'id': qa.get('id', f'query-{i}'),
                'question': qa['question'],
                'category': qa.get('category', 'unknown'),
                'label': qa.get('label', 'unknown'),
                'base_response': base_response,
                'finetuned_response': finetuned_response,
                'timestamp': datetime.now().isoformat()
            }
            
            # 응답 파일에 즉시 저장 (JSONL)
            with open(responses_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(response_data, ensure_ascii=False) + '\n')
            
            # 최종 JSON을 위해 메모리에도 저장
            all_responses.append(response_data)
            
            logger.info(f"  Saved responses for query {i+1}")
        
        # 최종 JSON 파일 저장
        logger.info("Saving final JSON file...")
        with open(final_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_responses, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Response generation completed!")
        logger.info(f"JSONL file: {responses_path}")
        logger.info(f"JSON file: {final_json_path}")
        logger.info(f"Total responses generated: {len(all_responses)}")
        
        return all_responses
    


def main():
    parser = argparse.ArgumentParser(description="Generate responses from base and fine-tuned models")
    parser.add_argument("--config", type=str, default="training_config.yaml", help="Training config file")
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-4B", help="Path to base model")
    parser.add_argument("--finetuned_model", type=str, required=True, help="Path to fine-tuned model")
    
    args = parser.parse_args()
    
    # 응답 생성 실행
    inference = ModelInference(args.config, args.base_model, args.finetuned_model)
    
    try:
        responses = inference.generate_responses()
        logger.info("Response generation completed successfully!")
        logger.info(f"Generated {len(responses)} response pairs")
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
