# 아주대학교 딥러닝 챌린지 발표 레포지토리 입니다.
aju_8000imgloader_largetoken.ipynb 파일을 코랩의 a100환경에서 실행하면 됩니다.

토큰이 너무 클 경우 aju_8000imgloader.ipynb 이 파일을 실행하셔도 됩니다.

## Multimodal Single-Model Pipeline (Qwen2-VL-7B + LoRA/QLoRA)

단일 멀티모달 모델(Qwen2-VL-7B-Instruct)에 **LoRA(QLoRA)**를 적용해 이미지 캡션 · VQA · 수학 추론 · In-context QA · 요약 5개 태스크를 태스크 분기 없이(프롬프트만) 처리하는 파이프라인입니다.
Colab A100 환경에서 학습/추론했고, 제출 형식(submission.csv)까지 자동 생성합니다.

Hugging Face Checkpoint: https://huggingface.co/tahn0321/qwen2vl-7b-ajou-lora

(베이스 모델은 Qwen/Qwen2-VL-7B-Instruct, 이 리포에는 LoRA 어댑터/프로세서만 포함됩니다.)

## 1. 핵심 특징

Single Model / Single Adapter / No Task Branching: if/else로 태스크 분기 금지 → 프롬프트만으로 태스크 유도

QLoRA(4-bit) + LoRA: 단일 GPU에서도 학습 가능한 저비용 튜닝

Vision Tower Frozen: 비전 인코더 동결, 디코더의 핵심 모듈(q/k/v/o, up/down/gate proj)만 LoRA

Auto-Grow Decoding: 종결부호 감지 후 이어 생성으로 장문/요약 중간 끊김 방지

견고한 데이터 로더: 이미지 URL 세션/재시도/UA + 디스크 캐시, 최대 변 448px 리사이즈

라벨 마스킹: 프롬프트 토큰 -100 마스킹, 정답 토큰만 loss

## Checkpoint

Hugging Face(Model repo): https://huggingface.co/tahn0321/qwen2vl-7b-ajou-lora

포함(예시): adapter_model.safetensors, adapter_config.json, processor_config.json 등


## 2. 사용법(어댑터 로드 - 파이썬)
``` python3
from transformers import AutoProcessor, AutoModelForCausalLM
from peft import PeftModel

base_id = "Qwen/Qwen2-VL-7B-Instruct"
adapter_id = "tahn0321/qwen2vl-7b-ajou-lora"

processor = AutoProcessor.from_pretrained(base_id, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(base_id, device_map="auto", trust_remote_code=True)
model = PeftModel.from_pretrained(base, adapter_id)
model.eval()

# 멀티모달 프롬프트 예시 (이미지 + 질문)
 messages = [
   {"role":"system","content":[{"type":"text","text":"You are a concise, honest, multimodal assistant."}]},
   {"role":"user","content":[{"type":"image"}, {"type":"text","text":"Question: ..."}]}
 ]
 prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
 enc = processor(text=prompt, images=[pil_image], return_tensors="pt")
 out = model.generate(**{k:v.to(model.device) for k,v in enc.items()}, max_new_tokens=192)
```
## 3. 환경 & 설치

CUDA 12.1 계열(PyTorch 공식 cu121 휠) 기준 예시입니다. CPU/다른 CUDA 버전이면 버전을 조정하세요.

권장: 가상환경 생성 후
```python
pip install --upgrade pip
pip install "torch==2.3.1+cu121" "torchvision==0.18.1+cu121" --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.46.0" "accelerate>=0.33.0" "peft>=0.11.1" \
           "bitsandbytes==0.43.3" pillow einops pandas datasets qwen-vl-utils
```
## 4. Quick Inference (submission.csv 생성)

테스트셋(deep_chal_multitask_dataset_test.parquet)을 이용해 submission.csv를 생성합니다.
레포에 포함된 Colab 스크립트(또는 동일 내용을 담은 train_and_infer_qwen2vl_lora.py)의 하단 추론 블록을 실행하면 됩니다.

## 예시
```python
python -u src/train_and_infer_qwen2vl_lora.py \
  --mode infer \
  --base_model Qwen/Qwen2-VL-7B-Instruct \
  --hf_lora_repo tahn0321/qwen2vl-7b-ajou-lora \
  --csv_test /path/to/deep_chal_multitask_dataset_test.parquet \
  --out_dir ./outputs \
  --submission ./outputs/submission.csv \
  --load_in_4bit True --bf16 True
```

출력: ./outputs/submission.csv (id,output)

추론 중 부분 저장: submission.csv.partial

디코딩: no_repeat_ngram_size=4, repetition_penalty=1.05 + Auto-Grow 이어 생성

스크립트를 노트북으로 바로 쓰는 경우, 상단 환경 변수와 경로(CSV_TEST, SAVE_DIR)만 맞춘 뒤 마지막 “추론” 섹션을 실행하세요.

## 5. Training

학습셋: deep_chal_multitask_dataset.parquet

검증셋(샘플): deep_chal_multitask_dataset_sample.parquet

스크립트 상단의 사용자 설정(경로/하이퍼)을 맞춘 뒤 학습 섹션을 실행합니다.
Colab 24h 제한 대응을 위해 23h 타임리밋 + 30분 주기 저장 콜백을 포함합니다.

## 옵션

USE_TRAIN_SUBSET=True, SUBSET_SIZE=8000 → 층화 8k 부분 학습(재현성 위해 인덱스 저장)

Vision Tower 동결, LoRA 대상 모듈: q_proj,k_proj,v_proj,o_proj,up_proj,down_proj,gate_proj

LoRA 하이퍼(예시): r=32, alpha=16, dropout=0.05, lr=1e-4, GA=16, per_device_batch=1, cosine, warmup=3%

## 6. 설계 메모

프롬프트 통일: 시스템 프롬프트 고정, 사용자 입력은 (이미지+질문) 또는 (지문 + Question:)

라벨 마스킹: 프롬프트 토큰은 -100으로 마스크, 정답 토큰만 loss

HTTP 로더 견고화: requests.Session + Retry/UA, 디스크 캐시, 최대 변 448px 리사이즈(패치 배수 정렬)

Auto-Grow: 태스크별 길이 프로파일(VQA/Caption/DocQA 등)을 두고 종결부호 감지 시까지 max_new_tokens를 점증


## 7. 참고

Base: Qwen/Qwen2-VL-7B-Instruct

라이브러리: PyTorch, Transformers, Accelerate, PEFT(LoRA), bitsandbytes

멀티모달 프롬프트 아이디어: Visual Instruction Tuning 계열

학습/추론 스크립트는 Colab A100 기준으로 검증
