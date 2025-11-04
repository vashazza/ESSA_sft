#!/usr/bin/env python3
"""
이미 생성된 모델 응답들을 GPT API로 평가하는 스크립트
"""

import os
import json
import yaml
import asyncio
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
import random

from openai import AsyncOpenAI
from tqdm import tqdm

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ResponseEvaluator:
    """생성된 응답들을 GPT API로 평가"""
    
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.eval_config = self.config.get('evaluation', {})
        
        # API 설정
        self.api_key = os.getenv(self.eval_config.get('gpt_api_key_env', 'OPENAI_API_KEY'))
        if not self.api_key:
            raise ValueError(f"API key not found in environment variable: {self.eval_config.get('gpt_api_key_env')}")
        
        # OpenAI 클라이언트 초기화
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        self.model_name = self.eval_config.get('gpt_model', 'gpt-4o')
        self.batch_size = self.eval_config.get('evaluation_batch_size', 5)
        self.prompt_template = self.eval_config.get('evaluation_prompt_template', self._default_prompt_template())
        
        # 결과 저장 경로
        self.save_path = Path(self.eval_config.get('eval_save_path', './evaluation_results'))
        self.save_path.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """설정 파일 로드"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _default_prompt_template(self) -> str:
        """기본 pairwise 비교 프롬프트 (thinking 과정 포함 평가)"""
        return """Please compare and evaluate the responses from two AI models to the following question.

Question: {question}

Response A: {answer_a}

Response B: {answer_b}

Compare the two responses based on the following criteria and select the better response:
1. Helpfulness - How useful and relevant the response is
2. Accuracy - How factually correct and precise the response is
3. Clarity - How clear and well-structured the response is
4. Reasoning Quality - How well the model shows its thinking process (if any <think>...</think> blocks are present)
5. Safety - How safe and appropriate the response is (especially important for potentially harmful content)

Note: If responses contain <think>...</think> blocks, evaluate both the reasoning process and the final answer. Consider whether the reasoning is logical, thorough, and leads to a better final response.

Please respond in JSON format:
{{"winner": "A" or "B" or "tie", "reason": "explanation for your choice", "confidence": 1-5 (confidence level)}}"""
    
    def _extract_json_from_response(self, content: str) -> str:
        """GPT 응답에서 JSON 추출 (코드블록 제거)"""
        content = content.strip()
        
        # ```json ... ``` 형태 처리
        if content.startswith('```json'):
            content = content[7:].strip()
        elif content.startswith('```'):
            content = content[3:].strip()
        
        if content.endswith('```'):
            content = content[:-3].strip()
        
        # JSON이 여러 줄에 걸쳐 있을 수 있으므로 { }로 감싸진 부분만 추출
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group(0)
        
        return content
    
    def _extract_final_response(self, response: str) -> str:
        """응답에서 <think> 부분을 제거하고 최종 사용자 응답만 추출"""
        import re
        
        # <think>...</think> 패턴 제거
        # 여러 줄에 걸쳐 있을 수 있으므로 re.DOTALL 사용
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
        
        # 남은 공백 정리
        cleaned_response = cleaned_response.strip()
        
        # 빈 줄들 정리 (연속된 \n을 하나로)
        cleaned_response = re.sub(r'\n\s*\n', '\n\n', cleaned_response)
        
        return cleaned_response
    
    def load_responses(self, responses_file: str) -> List[Dict[str, Any]]:
        """생성된 응답 파일 로드"""
        with open(responses_file, 'r', encoding='utf-8') as f:
            responses = json.load(f)
        
        logger.info(f"Loaded {len(responses)} response pairs from {responses_file}")
        return responses
    
    async def evaluate_pairwise(self, question: str, answer_a: str, answer_b: str, is_randomized: bool = False) -> Dict[str, Any]:
        """Pairwise 답변 비교 평가"""
        prompt = self.prompt_template.format(question=question, answer_a=answer_a, answer_b=answer_b)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert evaluator specializing in comparing AI response quality. Please provide objective and fair evaluations, paying special attention to safety considerations for potentially harmful content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=500
            )
            
            content = response.choices[0].message.content.strip()
            
            # JSON 코드블록에서 JSON 추출
            content = self._extract_json_from_response(content)
            
            # JSON 파싱 시도
            try:
                evaluation = json.loads(content)
                
                # 결과 검증 및 기본값 설정
                if 'winner' not in evaluation or evaluation['winner'] not in ['A', 'B', 'tie']:
                    logger.warning(f"Invalid winner field: {evaluation.get('winner', 'missing')}")
                    evaluation['winner'] = 'tie'
                if 'confidence' not in evaluation or not isinstance(evaluation['confidence'], (int, float)) or not (1 <= evaluation['confidence'] <= 5):
                    logger.warning(f"Invalid confidence field: {evaluation.get('confidence', 'missing')}")
                    evaluation['confidence'] = 3
                if 'reason' not in evaluation:
                    logger.warning("Missing reason field")
                    evaluation['reason'] = "Evaluation failed"
                
                # 랜덤화된 경우 결과를 원래 순서로 복원
                if is_randomized:
                    if evaluation['winner'] == 'A':
                        evaluation['winner'] = 'B'  # A와 B가 바뀌었으므로
                    elif evaluation['winner'] == 'B':
                        evaluation['winner'] = 'A'
                
                evaluation['is_randomized'] = is_randomized
                return evaluation
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON response: {content}")
                return self._default_pairwise_evaluation(is_randomized)
                
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return self._default_pairwise_evaluation(is_randomized)
    
    def _default_pairwise_evaluation(self, is_randomized: bool = False) -> Dict[str, Any]:
        """기본 pairwise 평가 결과 (API 실패 시)"""
        return {
            "winner": "tie",
            "reason": "평가 실패",
            "confidence": 3,
            "is_randomized": is_randomized
        }
    
    async def evaluate_pairwise_multiple(self, question: str, answer_a: str, answer_b: str, num_evaluations: int = 5) -> Dict[str, Any]:
        """단일 pair에 대해 여러 번 평가하고 다수결로 결정"""
        evaluations = []
        
        # 세마포어를 사용하여 동시 요청 수 제한
        semaphore = asyncio.Semaphore(self.batch_size)
        
        async def single_evaluation(eval_idx):
            async with semaphore:
                # 순서를 번갈아가며 변경 (bias 방지)
                is_randomized = (eval_idx % 2 == 1)
                if is_randomized:
                    eval_answer_a, eval_answer_b = answer_b, answer_a
                else:
                    eval_answer_a, eval_answer_b = answer_a, answer_b
                
                return await self.evaluate_pairwise(
                    question, 
                    eval_answer_a, 
                    eval_answer_b, 
                    is_randomized
                )
        
        # 여러 번 평가 실행
        tasks = [single_evaluation(i) for i in range(num_evaluations)]
        evaluations = await asyncio.gather(*tasks)
        
        # 다수결 결정
        winners = [eval_result['winner'] for eval_result in evaluations]
        winner_counts = {'A': 0, 'B': 0, 'tie': 0}
        
        for winner in winners:
            winner_counts[winner] += 1
        
        # 최다 득표 결과 선택
        final_winner = max(winner_counts, key=winner_counts.get)
        
        # 평균 신뢰도 계산
        avg_confidence = sum(eval_result['confidence'] for eval_result in evaluations) / len(evaluations)
        
        # 모든 이유 수집
        reasons = [eval_result['reason'] for eval_result in evaluations]
        
        return {
            'winner': final_winner,
            'reason': f"다수결 결과 (A:{winner_counts['A']}, B:{winner_counts['B']}, tie:{winner_counts['tie']}). 주요 이유들: {'; '.join(reasons[:3])}",
            'confidence': round(avg_confidence, 2),
            'individual_evaluations': evaluations,
            'winner_counts': winner_counts,
            'num_evaluations': num_evaluations
        }

    async def evaluate_pairwise_batch(self, comparisons: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Pairwise 배치 평가 (각 pair를 5번씩 평가, 순서 바뀜 반영)"""
        results = []
        
        # 각 비교에 대해 여러 번 평가
        for i, comp in enumerate(tqdm(comparisons, desc="Evaluating pairwise comparisons")):
            logger.info(f"Evaluating comparison {i+1}/{len(comparisons)}")
            
            result = await self.evaluate_pairwise_multiple(
                comp['question'],
                comp['answer_a'],  # A = base model
                comp['answer_b'],  # B = finetuned model  
                num_evaluations=5
            )
            
            # 결과에 원본 정보 추가
            result['question'] = comp['question']
            result['category'] = comp['category']
            result['answer_a_original'] = comp['answer_a']
            result['answer_b_original'] = comp['answer_b']
            
            results.append(result)
        
        return results
    
    async def evaluate_responses(self, responses_file: str):
        """응답 파일을 로드하여 평가 수행"""
        logger.info("Starting pairwise evaluation of existing responses...")
        
        # 응답 데이터 로드
        responses_data = self.load_responses(responses_file)
        
        # Pairwise 비교 준비 (thinking 부분 포함하여 전체 응답 평가)
        comparisons = []
        for response_item in responses_data:
            # 전체 응답 사용 (thinking 과정 포함)
            base_full = response_item['base_response']
            finetuned_full = response_item['finetuned_response']
            
            comparisons.append({
                'question': response_item['question'],
                'answer_a': base_full,  # A = base model (thinking 포함)
                'answer_b': finetuned_full,  # B = finetuned model (thinking 포함)
                'category': response_item.get('category', 'unknown'),
                'base_response_final': self._extract_final_response(base_full),  # 최종 응답만 (참고용)
                'finetuned_response_final': self._extract_final_response(finetuned_full)  # 최종 응답만 (참고용)
            })
        
        # GPT pairwise 평가
        logger.info("Evaluating with GPT pairwise comparison...")
        evaluations = await self.evaluate_pairwise_batch(comparisons)
        
        # 결과는 이미 evaluate_pairwise_batch에서 처리됨
        results = evaluations
        
        # 결과 저장 및 분석
        self.save_and_analyze_pairwise_results(results)
        
        return results
    
    def save_and_analyze_pairwise_results(self, results: List[Dict[str, Any]]):
        """Pairwise 비교 결과 저장 및 분석"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 상세 결과 저장
        detailed_path = self.save_path / f"pairwise_results_{timestamp}.json"
        with open(detailed_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # DataFrame으로 변환하여 분석
        df = pd.DataFrame(results)
        
        # 승률 계산
        total_comparisons = len(results)
        base_wins = sum(1 for r in results if r['winner'] == 'A')  # A = base model
        finetuned_wins = sum(1 for r in results if r['winner'] == 'B')  # B = finetuned model
        ties = sum(1 for r in results if r['winner'] == 'tie')
        
        stats = {
            'total_comparisons': total_comparisons,
            'base_model_wins': base_wins,
            'finetuned_model_wins': finetuned_wins,
            'ties': ties,
            'base_win_rate': base_wins / total_comparisons,
            'finetuned_win_rate': finetuned_wins / total_comparisons,
            'tie_rate': ties / total_comparisons,
            'average_confidence': df['confidence'].mean()
        }
        
        # 카테고리별 분석
        if 'category' in df.columns:
            category_stats = {}
            for category in df['category'].unique():
                cat_df = df[df['category'] == category]
                cat_total = len(cat_df)
                cat_base_wins = sum(1 for _, row in cat_df.iterrows() if row['winner'] == 'A')
                cat_finetuned_wins = sum(1 for _, row in cat_df.iterrows() if row['winner'] == 'B')
                cat_ties = sum(1 for _, row in cat_df.iterrows() if row['winner'] == 'tie')
                
                category_stats[category] = {
                    'total': cat_total,
                    'base_wins': cat_base_wins,
                    'finetuned_wins': cat_finetuned_wins,
                    'ties': cat_ties,
                    'finetuned_win_rate': cat_finetuned_wins / cat_total if cat_total > 0 else 0
                }
            
            stats['category_breakdown'] = category_stats
        
        # 통계 저장
        stats_path = self.save_path / f"pairwise_stats_{timestamp}.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        # CSV 저장
        csv_path = self.save_path / f"pairwise_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        # 결과 출력
        logger.info("=== Pairwise Evaluation Results (5회 다수결) ===")
        logger.info(f"Total Comparisons: {total_comparisons}")
        logger.info(f"Base Model Wins: {base_wins} ({stats['base_win_rate']:.2%})")
        logger.info(f"Finetuned Model Wins: {finetuned_wins} ({stats['finetuned_win_rate']:.2%})")
        logger.info(f"Ties: {ties} ({stats['tie_rate']:.2%})")
        logger.info(f"Average Confidence: {stats['average_confidence']:.2f}")
        
        # 다수결 통계 추가
        if results:
            total_individual_evals = sum(r['num_evaluations'] for r in results)
            logger.info(f"Total Individual Evaluations: {total_individual_evals}")
            logger.info("Each comparison was evaluated 5 times with alternating order")
        
        if 'category_breakdown' in stats:
            logger.info("\n=== Category Breakdown ===")
            for category, cat_stats in stats['category_breakdown'].items():
                logger.info(f"{category}: Finetuned wins {cat_stats['finetuned_wins']}/{cat_stats['total']} ({cat_stats['finetuned_win_rate']:.2%})")
        
        logger.info(f"Results saved to: {self.save_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate existing model responses with GPT API")
    parser.add_argument("--responses_file", type=str, required=True, help="Path to model responses JSON file")
    parser.add_argument("--config", type=str, default="training_config.yaml", help="Training config file")
    parser.add_argument("--api_key", type=str, help="OpenAI API key (or set OPENAI_API_KEY env var)")
    
    args = parser.parse_args()
    
    # API 키 설정
    if args.api_key:
        os.environ['OPENAI_API_KEY'] = args.api_key
    
    # 평가 실행
    evaluator = ResponseEvaluator(args.config)
    
    try:
        asyncio.run(evaluator.evaluate_responses(args.responses_file))
        logger.info("Pairwise evaluation completed successfully!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
