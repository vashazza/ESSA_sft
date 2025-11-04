# teacher_runner.py
# Batch pipeline: read CSV, load SPEC per Domain/Task, build prompts from files,
# generate teacher responses with Qwen Instruct, and save results as JSON.

import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# tqdm (optional). If not installed, degrade gracefully.
try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **kwargs): return x
    def _tqdm_write(msg): print(msg)
    tqdm.write = _tqdm_write

from teacher_spec import load_spec_text, render_prompts, DEFAULT_SPEC_ROOT

# ----- Column detection -----
# ### 변경점: ID 열 후보 추가 ###
ID_COL_CANDIDATES = ["id", "ID"]
PROMPT_COL_CANDIDATES = ["prompt", "query", "input", "user_query", "question", "instruction", "text"]
DOMAIN_COL_CANDIDATES = ["domain", "Domain", "도메인"]
TASK_COL_CANDIDATES   = ["task", "Task", "태스크"]

def detect_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if col.lower() == c.lower():
                return col
    return None

# ----- Model utils -----
def load_qwen(model_id: str, device: Optional[str] = None, dtype: Optional[str] = None):
    if dtype is None:
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    else:
        model_dtype = getattr(torch, dtype)

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=model_dtype,
        device_map="auto" if (device in [None, "auto"] and torch.cuda.is_available()) else None,
    )
    if device and device not in ["auto"]:
        model = model.to(device)
    return model, tok

# --- Robust CSV reader -------------------------------------------------
import csv, io, re
import pandas as pd

def _balanced_chunks(file_path, encoding="utf-8"):
    """
    따옴표(") 균형이 맞을 때까지 라인을 합쳐 하나의 레코드 청크로 반환.
    \" 또는 "" 같은 이스케이프는 균형 계산에서 제외.
    """
    def unescaped_quote_parity(s: str) -> int:
        # \\" 제거 후 "" 제거 → 남은 " 개수의 짝수/홀수로 균형 판단
        tmp = re.sub(r'\\"', '', s)
        tmp = tmp.replace('""', '')
        return tmp.count('"') % 2  # 0이면 짝수(균형), 1이면 홀수(열림)

    buf = []
    parity = 0
    with open(file_path, "r", encoding=encoding, errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n")
            buf.append(line)
            parity ^= unescaped_quote_parity(line)
            if parity == 0:               # 따옴표 균형이 맞음 → 하나의 레코드로 간주
                yield "\n".join(buf)
                buf = []
        if buf:                            # 파일 끝에서 잔여 버퍼(비정상)도 일단 내보냄
            yield "\n".join(buf)

def _read_csv_balanced(csv_path: str) -> pd.DataFrame:
    """
    균형 청크를 csv.reader로 파싱 → DataFrame으로 변환
    pandas가 못 읽는 깨진 행도 최대한 보존해서 읽어냄.
    """
    chunks = _balanced_chunks(csv_path, encoding="utf-8")
    header = None
    records = []

    for i, chunk in enumerate(chunks):
        reader = csv.reader(io.StringIO(chunk),
                            delimiter=",", quotechar='"',
                            doublequote=True, escapechar='\\')
        for row in reader:
            if header is None:
                header = [h.strip() for h in row]
                continue
            # 행 길이 불일치 시 패딩/트림
            if len(row) < len(header):
                row += [''] * (len(header) - len(row))
            elif len(row) > len(header):
                row = row[:len(header)]
            records.append(dict(zip(header, row)))

    if header is None:  # 빈 파일 등
        return pd.DataFrame()

    return pd.DataFrame.from_records(records, columns=header)

def safe_read_csv(csv_path: str) -> pd.DataFrame:
    """
    1차: 기존 설정으로 pandas로 읽기
    2차: 실패하면 균형 청크 파서로 재시도(데이터 최대한 보존)
    마지막: 그래도 실패하면 문제행 스킵해서라도 읽기(on_bad_lines="skip")
    """
    try:
        return pd.read_csv(
            csv_path,
            sep=",",
            quotechar='"',
            doublequote=True,
            escapechar='\\',
            engine="python",
            encoding="utf-8",
            dtype=str,
            keep_default_na=False
        )
    except pd.errors.ParserError as e:
        tqdm.write(f"[WARN] ParserError: {e}. Falling back to balanced parser.")
        try:
            df = _read_csv_balanced(str(csv_path))
            # pandas가 자동 NaN 변환 안 하도록: 이미 문자열로 들어옴
            return df.astype(str)
        except Exception as e2:
            tqdm.write(f"[WARN] Balanced parser failed: {e2}. Falling back to skip bad lines.")
            return pd.read_csv(
                csv_path,
                sep=",",
                quotechar='"',
                doublequote=True,
                escapechar='\\',
                engine="python",
                encoding="utf-8",
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip"
            )


def generate_one(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int = 512) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_prompt},
    ]
    enc = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    outputs = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        temperature=0.2,
        top_p=0.9,
        repetition_penalty=1.1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
    )
    text = tokenizer.decode(outputs[0][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text

# ----- Main -----
def run_batch(csv_path: Path,
              out_path: Path,
              system_tpl_path: Path,
              user_tpl_path: Path,
              spec_root: Path,
              model_id: str,
              max_new_tokens: int,
              device: Optional[str] = None,
              dtype: Optional[str] = None,
              filter_col: Optional[str] = None,
              filter_val: Optional[str] = None,
              limit: Optional[int] = None):

    import csv
    df = safe_read_csv(csv_path)
    # ---- 필터 & 리미트 적용 (있을 때만) ----
    if filter_col and filter_col in df.columns:
        if filter_val is not None:
            df = df[df[filter_col].astype(str) == str(filter_val)]
    if limit is not None:
        df = df.head(limit)

    if len(df) == 0:
        tqdm.write("[WARN] No rows to process after filtering/limit. Exiting.")
        # ### 변경점: 빈 JSON 배열 파일 생성 ###
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("[]")
        return

    # ### 변경점: ID와 프롬프트 열 감지 및 예외 처리 ###
    id_col = detect_column(df, ID_COL_CANDIDATES)
    if not id_col:
        raise ValueError(f"Could not find an ID column. Expected one of: {ID_COL_CANDIDATES}. Found: {list(df.columns)}")
        
    prompt_col = detect_column(df, PROMPT_COL_CANDIDATES)
    if not prompt_col:
        raise ValueError(f"Could not find a prompt column. Expected one of: {PROMPT_COL_CANDIDATES}. Found: {list(df.columns)}")

    domain_col = detect_column(df, DOMAIN_COL_CANDIDATES)
    task_col   = detect_column(df, TASK_COL_CANDIDATES)

    system_tpl = Path(system_tpl_path).read_text(encoding="utf-8")
    user_tpl   = Path(user_tpl_path).read_text(encoding="utf-8")

    model, tokenizer = load_qwen(model_id=model_id, device=device, dtype=dtype)

    responses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Teacher responses", unit="row"):
        query  = str(row[prompt_col])
        
        domain = str(row[domain_col]) if domain_col else "General"
        task   = str(row[task_col])   if task_col   else "Communication"

        try:
            spec_bullets = load_spec_text(domain=domain, task=task, spec_root=spec_root)
        except Exception as e:
            spec_bullets = "- (SPEC not found for this Domain/Task)"
            tqdm.write(f"[WARN] {e}")

        system_prompt, user_prompt = render_prompts(system_tpl, user_tpl,
                                                    domain=domain, task=task,
                                                    spec_bullets=spec_bullets, query=query)

        text = generate_one(model, tokenizer, system_prompt, user_prompt, max_new_tokens=max_new_tokens)
        responses.append(text)

    # ### 변경점: 최종 출력 데이터프레임 재구성 및 JSON으로 저장 ###
    # 필요한 열만 선택하여 새로운 데이터프레임 생성
    df_out = pd.DataFrame({
        "id": df[id_col],
        "prompt": df[prompt_col],
        "teacher_response": responses
    })
    
    # JSON 파일로 저장
    # orient='records'는 [{column: value, ...}, ...] 형태의 리스트로 만듭니다.
    # indent=2는 가독성을 위해 들여쓰기를 추가합니다.
    # force_ascii=False는 한글 등 비ASCII 문자가 깨지지 않도록 합니다.
    df_out.to_json(out_path, orient='records', indent=2, force_ascii=False)
    print(f"[DONE] Wrote: {out_path}")

def parse_args():
    ap = argparse.ArgumentParser(description="Run Teacher Model on a CSV of queries with SPEC-grounded quoting and save as JSON.")
    ap.add_argument("--csv", type=Path, required=True, help="Input CSV path (e.g., /home/elicer/workspace/1mo/data/Telecom/T_FR.csv)")
    # ### 변경점: 출력 파일명 및 도움말 수정 ###
    ap.add_argument("--out", type=Path, default=Path("teacher_outputs.json"), help="Output JSON path")
    ap.add_argument("--system_tpl", type=Path, default=Path("templates/teacher_system.txt"))
    ap.add_argument("--user_tpl",   type=Path, default=Path("templates/teacher_user.txt"))
    ap.add_argument("--spec_root",  type=Path, default=DEFAULT_SPEC_ROOT, help="Root dir of SPEC (Domain/Task JSONs)")
    ap.add_argument("--model",      type=str,   default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--max_new_tokens", type=int, default=4096)
    ap.add_argument("--device",     type=str,   default=None, help='e.g., "cuda", "cpu", or "auto"')
    ap.add_argument("--dtype",      type=str,   default=None, help='e.g., "bfloat16", "float16", "float32"')
    ap.add_argument("--filter_col", type=str, default=None, help="Column to filter (e.g., falsereject)")
    ap.add_argument("--filter_val", type=str, default=None, help='Keep rows where str(value)==filter_val (e.g., "1")')
    ap.add_argument("--limit",      type=int, default=None, help="Process only this many rows after filtering")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_batch(csv_path=args.csv,
              out_path=args.out,
              system_tpl_path=args.system_tpl,
              user_tpl_path=args.user_tpl,
              spec_root=args.spec_root,
              model_id=args.model,
              max_new_tokens=args.max_new_tokens,
              device=args.device,
              dtype=args.dtype,
              filter_col=args.filter_col,
              filter_val=args.filter_val,
              limit=args.limit)

