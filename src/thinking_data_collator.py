#!/usr/bin/env python3
"""
Thinking 모델을 위한 데이터 콜레이터
전체 시퀀스를 학습하여 추론 능력 향상
"""

import logging
from typing import Dict, Any
from transformers import PreTrainedTokenizer, DataCollatorForLanguageModeling

logger = logging.getLogger(__name__)


class ThinkingDataCollator(DataCollatorForLanguageModeling):
    """
    Thinking 모델을 위한 데이터 콜레이터
    사용자 질문은 마스킹하고, Assistant 응답(thinking + 답변)만 학습
    """
    
    def __init__(self, tokenizer: PreTrainedTokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
        self.assistant_token = "Assistant:"
        self.assistant_token_ids = tokenizer.encode(self.assistant_token, add_special_tokens=False)
        
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        
        if "labels" in batch:
            labels = batch["labels"].clone()
            
            for i, label_seq in enumerate(labels):
                # Assistant: 이전 부분(사용자 질문)을 -100으로 마스킹
                assistant_start = self._find_assistant_start(label_seq)
                if assistant_start is not None:
                    labels[i, :assistant_start] = -100
                    
            batch["labels"] = labels
            
        return batch
    
    def _find_assistant_start(self, input_ids):
        """Assistant: 토큰의 시작 위치 찾기"""
        if len(self.assistant_token_ids) == 0:
            return None
            
        input_list = input_ids.tolist()
        
        for i in range(len(input_list) - len(self.assistant_token_ids) + 1):
            if input_list[i:i+len(self.assistant_token_ids)] == self.assistant_token_ids:
                return i + len(self.assistant_token_ids)
                
        return None


def create_data_collator(tokenizer: PreTrainedTokenizer, config: Dict[str, Any]) -> DataCollatorForLanguageModeling:
    """
    Thinking 모델을 위한 데이터 콜레이터 생성
    
    사용자 질문은 마스킹하고, Assistant 응답(thinking + 답변)만 학습하여
    효율적으로 추론 과정을 배우도록 함
    """
    logger.info("Creating data collator for Assistant response learning (including thinking process)")
    
    return ThinkingDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8,  # 효율성을 위한 패딩
        return_tensors="pt"
    )
