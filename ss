import os, torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel

MODEL_ID = os.getenv("MODEL_ID", "jinaai/jina-embeddings-v2-base-en")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
mdl = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE).eval()

app = FastAPI()

class Req(BaseModel):
    input: str | list[str]

def pool(x, m):
    m = m.unsqueeze(-1).to(x.dtype)
    return (x*m).sum(1) / m.sum(1).clamp(min=1e-9)

@app.post("/v1/embeddings")
def emb(r: Req):
    texts = r.input if isinstance(r.input, list) else [r.input]
    b = tok(texts, padding=True, truncation=True, return_tensors="pt")
    b = {k: v.to(DEVICE) for k, v in b.items()}
    with torch.no_grad():
        e = pool(mdl(**b).last_hidden_state, b["attention_mask"])
        e = torch.nn.functional.normalize(e, p=2, dim=1)
    return {"data": [{"index": i, "embedding": e[i].tolist()} for i in range(e.size(0))], "model": MODEL_ID}
