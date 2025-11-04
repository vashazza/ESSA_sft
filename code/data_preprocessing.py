#!/usr/bin/env python3
"""
SFT 데이터 전처리 통합 스크립트
- JSON 형식 SFT 데이터를 Qwen3 Thinking 형식으로 변환
- 처리된 데이터를 훈련용과 검증용으로 분할 및 합치기
"""

import os
import json
import logging
import re
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class QwenDataProcessor:
    """Qwen3 데이터 처리 유틸리티"""
    
    def format_thinking_response(self, instruction: str, thinking: str, output: str) -> Dict[str, Any]:
        """Qwen3 Thinking 형식으로 응답 포맷팅 (공식 권장사항 반영)"""
        # Thinking 모델을 위해 <think> 태그로 추론 과정을 감싸기
        if thinking:
            full_response = f"<think>\n{thinking}\n</think>\n\n{output}"
        else:
            full_response = output
        
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": full_response}
        ]
        
        return {
            "messages": messages,
            "instruction": instruction,
            "thinking": thinking,
            "output": output,
            "full_response": full_response
        }
    
    def save_dataset(self, data: List[Dict], output_path: str, format_type: str = "jsonl"):
        """데이터셋 저장"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format_type == "jsonl":
            with open(output_path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        elif format_type == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(data)} items to {output_path}")


class SFTDataProcessor:
    """SFT 데이터 형식 전용 처리기"""
    
    def __init__(self):
        self.processor = QwenDataProcessor()
        
    def parse_teacher_response(self, teacher_response: str) -> Tuple[str, str]:
        """teacher_response 필드에서 ##thought와 ##response 부분 분리"""
        if not teacher_response or not isinstance(teacher_response, str):
            return "", ""
        
        # ##thought 부분 추출 (thinking/reasoning)
        thought_match = re.search(r'##thought\s*(.*?)(?=##response|$)', teacher_response, re.DOTALL | re.IGNORECASE)
        thinking = thought_match.group(1).strip() if thought_match else ""
        
        # ##response 부분 추출 (final answer)
        response_match = re.search(r'##response\s*(.*?)$', teacher_response, re.DOTALL | re.IGNORECASE)
        final_response = response_match.group(1).strip() if response_match else ""
        
        # 만약 분리가 안 되면 전체를 final_response로 사용
        if not thinking and not final_response:
            final_response = teacher_response.strip()
            thinking = ""
        
        return thinking, final_response
    
    def analyze_json_file(self, file_path: str) -> Dict[str, Any]:
        """JSON 파일 분석"""
        logger.info(f"Analyzing file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {file_path}: {e}")
            return {"file": file_path, "count": 0, "sample": None, "keys": [], "error": str(e)}
        
        if not data or not isinstance(data, list):
            return {"file": file_path, "count": 0, "sample": None, "keys": []}
        
        # 기본 통계
        analysis = {
            "file": file_path,
            "count": len(data),
            "sample": data[0] if data else None,
            "keys": list(data[0].keys()) if data else [],
        }
        
        # teacher_response 필드 분석
        if data and "teacher_response" in data[0]:
            valid_responses = 0
            thought_parts = 0
            response_parts = 0
            
            for item in data[:100]:  # 샘플링해서 분석
                teacher_response = item.get("teacher_response", "")
                if teacher_response and teacher_response.strip():
                    valid_responses += 1
                    thinking_part, final_part = self.parse_teacher_response(teacher_response)
                    if thinking_part:
                        thought_parts += 1
                    if final_part:
                        response_parts += 1
            
            analysis["teacher_response_analysis"] = {
                "valid_responses": valid_responses,
                "has_thought": thought_parts,
                "has_response": response_parts,
                "sample_size": min(100, len(data))
            }
        
        return analysis
    
    def process_json_format(self, data: List[Dict]) -> List[Dict]:
        """JSON 형식 데이터를 Qwen3 thinking 형식으로 변환"""
        processed_data = []
        
        for item in tqdm(data, desc="Processing items"):
            try:
                # 필수 필드 확인
                prompt = item.get('prompt', '').strip()
                teacher_response = item.get('teacher_response', '').strip()
                
                if not prompt:
                    logger.warning(f"Skipping item with missing prompt: {item.get('id', 'unknown')}")
                    continue
                
                if not teacher_response:
                    logger.warning(f"Skipping item with missing teacher_response: {item.get('id', 'unknown')}")
                    continue
                
                # teacher_response에서 ##cot와 ##response 부분 분리
                thinking_part, final_part = self.parse_teacher_response(teacher_response)
                
                if not final_part:
                    logger.warning(f"No final response found for item: {item.get('id', 'unknown')}")
                    continue
                
                # thinking 부분 구성 (원본 내용 그대로 사용)
                thinking = thinking_part if thinking_part else ""
                
                # Qwen3 형식으로 변환
                formatted_item = self.processor.format_thinking_response(
                    instruction=prompt,
                    thinking=thinking,
                    output=final_part
                )
                
                # 메타데이터 추가
                formatted_item['metadata'] = {
                    'original_id': item.get('id'),
                    'source_file': Path(item.get('source_file', 'unknown')).name,
                    'has_thought': bool(thinking_part),
                    'original_teacher_response_length': len(teacher_response)
                }
                
                processed_data.append(formatted_item)
                
            except Exception as e:
                logger.error(f"Error processing item {item.get('id', 'unknown')}: {e}")
                continue
        
        return processed_data
    
    def process_nemotron_format(self, data: List[Dict]) -> List[Dict]:
        """Nemotron 형식 데이터를 Qwen3 thinking 형식으로 변환"""
        processed_data = []
        
        for item in tqdm(data, desc="Processing nemotron items"):
            try:
                # Nemotron 형식: messages 배열에서 user와 assistant 메시지 추출
                messages = item.get('messages', [])
                if len(messages) < 2:
                    logger.warning(f"Skipping item with insufficient messages: {len(messages)}")
                    continue
                
                user_message = None
                assistant_message = None
                
                for msg in messages:
                    if msg.get('role') == 'user':
                        user_message = msg.get('content', '').strip()
                    elif msg.get('role') == 'assistant':
                        assistant_message = msg.get('content', '').strip()
                
                if not user_message or not assistant_message:
                    logger.warning(f"Skipping item with missing user or assistant message")
                    continue
                
                # <think> 태그가 있는지 확인하여 thinking과 output 분리
                thinking = ""
                output = assistant_message
                
                if "<think>" in assistant_message and "</think>" in assistant_message:
                    # <think>...</think> 부분을 thinking으로, 나머지를 output으로 분리
                    import re
                    think_match = re.search(r'<think>(.*?)</think>', assistant_message, re.DOTALL)
                    if think_match:
                        thinking = think_match.group(1).strip()
                        # <think>...</think> 부분을 제거한 나머지를 output으로
                        output = re.sub(r'<think>.*?</think>', '', assistant_message, flags=re.DOTALL).strip()
                
                # Qwen3 형식으로 변환
                formatted_item = self.processor.format_thinking_response(
                    instruction=user_message,
                    thinking=thinking,
                    output=output
                )
                
                # 메타데이터 추가
                formatted_item['metadata'] = {
                    'source_file': 'nemotron_5000.json',
                    'has_thinking': bool(thinking),
                    'original_format': 'nemotron',
                    'meta': item.get('meta', {})
                }
                
                processed_data.append(formatted_item)
                
            except Exception as e:
                logger.error(f"Error processing nemotron item: {e}")
                continue
        
        return processed_data
    
    def is_nemotron_format(self, data: List[Dict]) -> bool:
        """데이터가 Nemotron 형식인지 확인"""
        if not data or not isinstance(data, list):
            return False
        
        # 첫 번째 항목을 확인하여 Nemotron 형식인지 판단
        sample = data[0]
        
        # Nemotron 형식: messages 키가 있고, 각 메시지에 role과 content가 있음
        if 'messages' in sample and isinstance(sample['messages'], list):
            messages = sample['messages']
            if len(messages) >= 2:
                # user와 assistant 메시지가 있는지 확인
                has_user = any(msg.get('role') == 'user' for msg in messages)
                has_assistant = any(msg.get('role') == 'assistant' for msg in messages)
                return has_user and has_assistant
        
        return False
    
    def process_single_file(self, input_file: str, output_dir: str = "processed_sft_data", 
                           preview_only: bool = False) -> Dict[str, Any]:
        """단일 파일 처리"""
        
        # 파일 분석
        analysis = self.analyze_json_file(input_file)
        
        print(f"\n{'='*60}")
        print(f"📁 File: {Path(input_file).name}")
        print(f"📊 Total items: {analysis['count']}")
        print(f"🔑 Keys: {analysis['keys']}")
        
        if "error" in analysis:
            print(f"❌ Error: {analysis['error']}")
            return analysis
        
        if analysis.get('sample'):
            print(f"\n📋 Sample item:")
            sample = analysis['sample']
            for key, value in sample.items():
                if key == 'teacher_response' and isinstance(value, str) and len(value) > 200:
                    # teacher_response 필드는 앞부분만 보여주기
                    print(f"  {key}: {value[:200]}...")
                elif isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
        if 'teacher_response_analysis' in analysis:
            stats = analysis['teacher_response_analysis']
            print(f"\n📈 Teacher response analysis (sample of {stats['sample_size']}):")
            print(f"  Valid responses: {stats['valid_responses']}")
            print(f"  Has ##thought part: {stats['has_thought']}")
            print(f"  Has ##response part: {stats['has_response']}")
        
        if preview_only:
            print(f"\n👀 Preview mode - no processing done")
            return analysis
        
        # 실제 처리
        print(f"\n🔄 Processing data...")
        
        # 데이터 로드
        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading {input_file}: {e}")
            return {**analysis, "error": str(e)}
        
        if not isinstance(data, list):
            logger.error(f"Expected list format in {input_file}")
            return {**analysis, "error": "Not a list format"}
        
        # 데이터 형식 감지 및 처리
        if self.is_nemotron_format(data):
            print(f"🔍 Nemotron 형식 감지됨")
            processed_data = self.process_nemotron_format(data)
        else:
            print(f"🔍 SFT 형식으로 처리")
            processed_data = self.process_json_format(data)
        
        # 출력 파일 경로 생성
        os.makedirs(output_dir, exist_ok=True)
        input_name = Path(input_file).stem
        output_file = os.path.join(output_dir, f"{input_name}_processed.jsonl")
        
        # 저장
        self.processor.save_dataset(processed_data, output_file, format_type="jsonl")
        
        print(f"✅ Processed {len(processed_data)}/{len(data)} items")
        print(f"💾 Saved to: {output_file}")
        
        # 처리 결과 샘플 출력
        if processed_data:
            print(f"\n📋 Processed sample:")
            sample = processed_data[0]
            print(f"  Instruction: {sample['instruction'][:100]}...")
            print(f"  Thinking: {sample['thinking'][:150]}...")
            print(f"  Output: {sample['output'][:100]}...")
        
        return {
            **analysis,
            "processed_count": len(processed_data),
            "output_file": output_file,
            "success_rate": len(processed_data) / len(data) if data else 0
        }
    
    def process_all_files(self, sft_data_dir: str = "sft_data", 
                         output_dir: str = "processed_sft_data",
                         preview_only: bool = False,
                         interactive: bool = True) -> Dict[str, Any]:
        """모든 JSON 파일 처리"""
        
        # SFT 데이터 디렉토리 확인
        if not os.path.exists(sft_data_dir):
            logger.error(f"SFT data directory not found: {sft_data_dir}")
            return {}
        
        # JSON 파일 찾기
        json_files = list(Path(sft_data_dir).glob("*.json"))
        
        if not json_files:
            logger.error(f"No JSON files found in {sft_data_dir}")
            return {}
        
        print(f"🔍 Found {len(json_files)} JSON files:")
        for i, file in enumerate(json_files, 1):
            print(f"  {i}. {file.name}")
        
        results = {}
        
        for file_path in json_files:
            if interactive:
                print(f"\n{'='*80}")
                response = input(f"Process {file_path.name}? (y/n/s=skip all remaining): ").lower().strip()
                
                if response == 's':
                    print("Skipping all remaining files...")
                    break
                elif response != 'y':
                    print(f"Skipping {file_path.name}")
                    continue
            
            try:
                result = self.process_single_file(
                    str(file_path), 
                    output_dir=output_dir,
                    preview_only=preview_only
                )
                results[file_path.name] = result
                
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")
                results[file_path.name] = {"error": str(e)}
        
        # 전체 요약
        print(f"\n{'='*80}")
        print(f"📊 PROCESSING SUMMARY")
        print(f"{'='*80}")
        
        total_items = 0
        total_processed = 0
        
        for filename, result in results.items():
            if "error" in result:
                print(f"❌ {filename}: ERROR - {result['error']}")
            else:
                count = result.get('count', 0)
                processed = result.get('processed_count', 0)
                success_rate = result.get('success_rate', 0) * 100
                
                total_items += count
                total_processed += processed
                
                print(f"✅ {filename}: {processed}/{count} items ({success_rate:.1f}%)")
        
        if total_items > 0:
            overall_success = (total_processed / total_items) * 100
            print(f"\n🎯 OVERALL: {total_processed}/{total_items} items ({overall_success:.1f}%)")
        
        return results


class DataCombiner:
    """처리된 데이터를 합치고 분할하는 클래스"""
    
    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """JSONL 파일 로드"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line.strip()))
                except json.JSONDecodeError as e:
                    logger.warning(f"Error parsing line in {file_path}: {e}")
                    continue
        return data

    @staticmethod
    def save_jsonl(data: List[Dict], file_path: str):
        """JSONL 파일 저장"""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def combine_processed_data(self, processed_dir: str, output_dir: str, val_ratio: float = 0.1, 
                             seed: int = 42) -> Dict[str, Any]:
        """처리된 데이터 합치기"""
        
        # 처리된 파일 찾기
        processed_files = list(Path(processed_dir).glob("*_processed.jsonl"))
        
        if not processed_files:
            logger.error(f"No processed files found in {processed_dir}")
            return {}
        
        # 데이터 로드 및 합치기
        all_data = []
        file_stats = {}
        
        for file_path in processed_files:
            data = self.load_jsonl(str(file_path))
            if data:
                logger.info(f"Loaded {len(data)} items from {file_path.name}")
                file_stats[file_path.name] = len(data)
                all_data.extend(data)
        
        if not all_data:
            logger.error("No data loaded!")
            return {}
        
        # 데이터 섞기
        random.seed(seed)
        random.shuffle(all_data)
        
        # 훈련/검증 분할
        val_size = int(len(all_data) * val_ratio)
        val_data = all_data[:val_size]
        train_data = all_data[val_size:]
        
        # 저장
        train_path = os.path.join(output_dir, "combined_train.jsonl")
        val_path = os.path.join(output_dir, "combined_val.jsonl")
        
        self.save_jsonl(train_data, train_path)
        self.save_jsonl(val_data, val_path)
        
        # 통계
        stats = {
            "total_items": len(all_data),
            "train_items": len(train_data),
            "val_items": len(val_data),
            "file_stats": file_stats,
            "train_path": train_path,
            "val_path": val_path
        }
        
        # 요약 출력
        print(f"\n{'='*60}")
        print(f"📊 Data Combination Summary")
        print(f"{'='*60}")
        print(f"Total items: {stats['total_items']}")
        print(f"Training items: {stats['train_items']}")
        print(f"Validation items: {stats['val_items']}")
        print(f"\nFile contributions:")
        for fname, count in stats['file_stats'].items():
            print(f"  {fname}: {count} items")
        print(f"\n💾 Saved to:")
        print(f"  Train: {stats['train_path']}")
        print(f"  Val: {stats['val_path']}")
        
        return stats


class DataPreprocessingPipeline:
    """전체 데이터 전처리 파이프라인"""
    
    def __init__(self):
        self.processor = SFTDataProcessor()
        self.combiner = DataCombiner()
    
    def run_full_pipeline(self, sft_data_dir: str = "sft_data", 
                         output_dir: str = "processed_sft_data",
                         val_ratio: float = 0.1,
                         seed: int = 42,
                         interactive: bool = True,
                         preview_only: bool = False) -> Dict[str, Any]:
        """전체 파이프라인 실행"""
        
        print(f"🚀 Starting SFT Data Preprocessing Pipeline")
        print(f"{'='*80}")
        print(f"Input directory: {sft_data_dir}")
        print(f"Output directory: {output_dir}")
        print(f"Validation ratio: {val_ratio}")
        print(f"Random seed: {seed}")
        print(f"Interactive mode: {interactive}")
        print(f"Preview only: {preview_only}")
        
        # Step 1: 개별 파일 처리
        print(f"\n📝 Step 1: Processing individual JSON files...")
        processing_results = self.processor.process_all_files(
            sft_data_dir=sft_data_dir,
            output_dir=output_dir,
            preview_only=preview_only,
            interactive=interactive
        )
        
        if not processing_results or preview_only:
            return {"processing_results": processing_results}
        
        # Step 2: 처리된 데이터 합치기
        print(f"\n🔗 Step 2: Combining processed data...")
        combination_results = self.combiner.combine_processed_data(
            processed_dir=output_dir,
            output_dir=output_dir,
            val_ratio=val_ratio,
            seed=seed
        )
        
        # 최종 결과
        final_results = {
            "processing_results": processing_results,
            "combination_results": combination_results,
            "pipeline_success": bool(combination_results)
        }
        
        if combination_results:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"📈 Final Statistics:")
            print(f"  Total processed items: {combination_results['total_items']}")
            print(f"  Training samples: {combination_results['train_items']}")
            print(f"  Validation samples: {combination_results['val_items']}")
            print(f"  Ready for training: {combination_results['train_path']}")
            print(f"  Ready for validation: {combination_results['val_path']}")
        else:
            print(f"\n❌ Pipeline failed during combination step")
        
        return final_results


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="SFT 데이터 전처리 통합 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 전체 파이프라인 실행 (대화형)
  python data_preprocessing.py --mode pipeline
  
  # 전체 파이프라인 실행 (배치 모드)
  python data_preprocessing.py --mode pipeline --batch
  
  # 개별 파일만 처리
  python data_preprocessing.py --mode process --file path/to/file.json
  
  # 모든 파일 처리만 (합치기 제외)
  python data_preprocessing.py --mode process --sft_dir sft_data
  
  # 이미 처리된 데이터 합치기만
  python data_preprocessing.py --mode combine --processed_dir processed_sft_data
  
  # 미리보기만 (실제 처리 안함)
  python data_preprocessing.py --mode process --preview
        """
    )
    
    # 모드 선택
    parser.add_argument("--mode", type=str, choices=["pipeline", "process", "combine"], 
                       default="pipeline", help="실행 모드")
    
    # 공통 옵션
    parser.add_argument("--sft_dir", type=str, default="sft_data", 
                       help="SFT 데이터 디렉토리 경로")
    parser.add_argument("--output_dir", type=str, default="processed_general_only_data",
                       help="출력 디렉토리 경로")
    parser.add_argument("--batch", action="store_true",
                       help="배치 모드 (상호작용 없음)")
    
    # 처리 관련 옵션
    parser.add_argument("--file", type=str, help="특정 파일만 처리")
    parser.add_argument("--preview", action="store_true", 
                       help="미리보기만 (실제 처리 안함)")
    
    # 합치기 관련 옵션
    parser.add_argument("--processed_dir", type=str, help="처리된 데이터 디렉토리 (combine 모드용)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                       help="검증 데이터 비율 (0-1)")
    parser.add_argument("--seed", type=int, default=42,
                       help="랜덤 시드")
    
    args = parser.parse_args()
    
    # 모드별 실행
    if args.mode == "pipeline":
        # 전체 파이프라인 실행
        pipeline = DataPreprocessingPipeline()
        results = pipeline.run_full_pipeline(
            sft_data_dir=args.sft_dir,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed,
            interactive=not args.batch,
            preview_only=args.preview
        )
        
    elif args.mode == "process":
        # 처리만 실행
        processor = SFTDataProcessor()
        
        if args.file:
            # 단일 파일 처리
            if not os.path.exists(args.file):
                logger.error(f"File not found: {args.file}")
                return
            
            result = processor.process_single_file(
                args.file, 
                output_dir=args.output_dir,
                preview_only=args.preview
            )
        else:
            # 모든 파일 처리
            results = processor.process_all_files(
                sft_data_dir=args.sft_dir,
                output_dir=args.output_dir,
                preview_only=args.preview,
                interactive=not args.batch
            )
    
    elif args.mode == "combine":
        # 합치기만 실행
        combiner = DataCombiner()
        processed_dir = args.processed_dir or args.output_dir
        
        results = combiner.combine_processed_data(
            processed_dir=processed_dir,
            output_dir=args.output_dir,
            val_ratio=args.val_ratio,
            seed=args.seed
        )


if __name__ == "__main__":
    main()
