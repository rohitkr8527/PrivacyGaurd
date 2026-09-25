import json, os
from pathlib import Path
import modal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME="Qwen/Qwen3-4B"
MODEL_REVISION="1cfa9a7"
MODEL_DIR="/models/qwen3-4b"
ADAPTER_REPO="rohitkmr8527/privacyguard-qwen3-4b-qlora"
ADAPTER_DIR="/models/privacyguard-qwen3-4b-qlora"
LORA_RANK=16
MAX_MODEL_LEN=4096
MAX_NEW_TOKENS=384
GPU_MEMORY_UTILIZATION=0.88

ALLOWED_TYPES=frozenset({
"DATE","GIVENNAME","SURNAME","EMAIL","CITY","TITLE","TELEPHONENUM","AGE",
"STREET","BUILDINGNUM","ZIPCODE","IDCARDNUM","CREDITCARDNUMBER",
"DRIVERLICENSENUM","GENDER","TAXNUM","SEX","SOCIALNUM","PASSPORTNUM"
})

SYSTEM_PROMPT="""You are a PII extraction system.

Extract every personally identifiable information entity from the input text.

Use ONLY these entity types:
DATE, GIVENNAME, SURNAME, EMAIL, CITY, TITLE, TELEPHONENUM, AGE, STREET,
BUILDINGNUM, ZIPCODE, IDCARDNUM, CREDITCARDNUMBER, DRIVERLICENSENUM,
GENDER, TAXNUM, SEX, SOCIALNUM, PASSPORTNUM.

Rules:
1. Copy entity text exactly as it appears in the input.
2. Keep GIVENNAME and SURNAME separate.
3. Keep STREET and BUILDINGNUM separate.
4. Use only the entity types listed above.
5. Do not invent entities.
6. If no PII exists, return an empty entities list.
7. Return ONLY valid raw JSON.
8. Do not use Markdown.
9. Do not provide explanations.

Required format:
{"entities":[{"text":"exact substring","type":"TYPE"}]}"""

app=modal.App("privacyguard-demo")
hf_secret=modal.Secret.from_dotenv(PROJECT_ROOT, filename=".env")

def download_assets(model_name, model_revision, model_dir, adapter_repo, adapter_dir):
    from huggingface_hub import HfApi, snapshot_download
    token=os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is missing.")
    info=HfApi(token=token).model_info(repo_id=adapter_repo, token=token)
    if not info.sha:
        raise RuntimeError("Could not resolve adapter revision.")
    snapshot_download(repo_id=model_name, revision=model_revision, local_dir=model_dir, token=token)
    snapshot_download(repo_id=adapter_repo, revision=info.sha, local_dir=adapter_dir, token=token)
    config_path=Path(adapter_dir)/"adapter_config.json"
    cfg=json.loads(config_path.read_text(encoding="utf-8"))
    if cfg.get("r") != LORA_RANK:
        raise RuntimeError(f"Unexpected LoRA rank: {cfg.get('r')!r}")
    if cfg.get("base_model_name_or_path") in {model_name, model_dir, "/models/qwen3-4b"}:
        cfg["base_model_name_or_path"]=model_name
        config_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    else:
        raise RuntimeError("Adapter/base-model mismatch.")

image=(modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
       .entrypoint([])
       .uv_pip_install("vllm==0.21.0")
       .env({"TOKENIZERS_PARALLELISM":"false","HF_HUB_DISABLE_TELEMETRY":"1"})
       .run_function(download_assets,args=(MODEL_NAME,MODEL_REVISION,MODEL_DIR,ADAPTER_REPO,ADAPTER_DIR),
                     secrets=[hf_secret],cpu=4,memory=16384,timeout=3600))

_llm=None
_sampling=None
_lora=None

def parse_response(raw):
    try:
        obj=json.loads(raw.strip())
    except (json.JSONDecodeError,TypeError):
        return [],False
    if isinstance(obj,dict) and isinstance(obj.get("entities"),list):
        return obj["entities"],True
    return [],False

def align_entities(text, entities):
    out=[]; used=set()
    for e in entities:
        if not isinstance(e,dict): continue
        value=e.get("text"); label=e.get("type")
        if not isinstance(value,str) or not isinstance(label,str): continue
        value=value.strip(); label=label.strip().upper()
        if not value or label not in ALLOWED_TYPES: continue
        pos=text.find(value)
        while pos!=-1:
            key=(pos,pos+len(value),label)
            if key not in used:
                used.add(key)
                out.append({"text":value,"type":label,"start":pos,"end":pos+len(value)})
                break
            pos=text.find(value,pos+1)
    return sorted(out,key=lambda x:(x["start"],x["end"]))

def redact(text, entities):
    out=text
    for e in sorted(entities,key=lambda x:(x["start"],x["end"]),reverse=True):
        out=out[:e["start"]]+f"[{e['type']}]"+out[e["end"]:]
    return out

def load_model():
    global _llm,_sampling,_lora
    if _llm is not None:
        return
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    _llm=LLM(model=MODEL_DIR,tokenizer=MODEL_DIR,dtype="bfloat16",
             max_model_len=MAX_MODEL_LEN,gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
             enable_prefix_caching=True,enable_lora=True,max_lora_rank=LORA_RANK,max_loras=1)
    _sampling=SamplingParams(temperature=0.0,max_tokens=MAX_NEW_TOKENS,seed=42)
    _lora=LoRARequest("privacyguard-pii",1,ADAPTER_DIR)

@app.function(image=image,gpu="L4",cpu=4,memory=24576,timeout=900,scaledown_window=300)
def detect_and_redact(text:str):
    text=text.strip()
    if not text:
        raise ValueError("text must not be empty")
    if len(text)>12000:
        raise ValueError("Demo input is limited to 12,000 characters.")
    load_model()
    conv=[[{"role":"system","content":SYSTEM_PROMPT},
           {"role":"user","content":"Extract all PII from this text:\n\n"+text}]]
    output=_llm.chat(conv,sampling_params=_sampling,lora_request=_lora,use_tqdm=False,
                     chat_template_kwargs={"enable_thinking":False})[0].outputs[0]
    raw=output.text.strip()
    raw_entities,valid=parse_response(raw)
    entities=align_entities(text,raw_entities)
    return {"redacted_text":redact(text,entities),"entities":entities,"entity_count":len(entities),
            "json_valid":valid,"finish_reason":output.finish_reason or "unknown","raw_response":raw}
