from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch, torch.nn as nn
import numpy as np, json

app = FastAPI()

MODEL_PATH  = "../model/draftnet_roles_only.pt"
PROFILE_JSON= "../model/hero_profile_roles.json"

with open(PROFILE_JSON, "r", encoding="utf-8") as f:
    hero_profile = {int(k): np.array(v, dtype=np.float32)
                    for k, v in json.load(f).items()}

HERO_COUNT   = max(hero_profile.keys()) + 1
PROFILE_DIM  = len(next(iter(hero_profile.values())))
INPUT_DIM    = 2*HERO_COUNT + 2*PROFILE_DIM + 2

ROLE_IDX = {'Carry':0, 'Support':2}


role_matrix = np.stack([vec[3:] for vec in hero_profile.values()])

def agg_profile(ids):
    if not ids:
        return np.zeros(PROFILE_DIM, dtype=np.float32)
    return np.mean([hero_profile[h] for h in ids], axis=0)

def role_counts(ids):
    if not ids:
        return 0, 0
    carr = int(role_matrix[ids, ROLE_IDX['Carry']].sum())
    supp = int(role_matrix[ids, ROLE_IDX['Support']].sum())
    return carr, supp

class DraftNet(nn.Module):
    def __init__(self, input_dim, hero_count):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512), nn.SiLU(),
            nn.Linear(512, 256),       nn.SiLU(),
            nn.Linear(256, hero_count)
        )
    def forward(self, x, mask):
        logits = self.layers(x)
        return logits.masked_fill(mask.bool(), -65504.0)

model = DraftNet(INPUT_DIM, HERO_COUNT)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

class DraftInput(BaseModel):
    my_picks:  List[int]
    opp_picks: List[int]

def build_x(my, opp):
    n_carr, n_supp = role_counts(my)
    x = np.zeros(INPUT_DIM, dtype=np.float32)

    if my:  x[np.asarray(my,  dtype=np.int64)] = 1.
    if opp: x[HERO_COUNT + np.asarray(opp, dtype=np.int64)] = 1.

    x[2*HERO_COUNT : 2*HERO_COUNT+PROFILE_DIM]               = agg_profile(my)
    x[2*HERO_COUNT+PROFILE_DIM : 2*HERO_COUNT+2*PROFILE_DIM] = agg_profile(opp)

    x[-2] = n_carr
    x[-1] = n_supp
    return x

@app.post("/recommendations")
def recommend_hero(data: DraftInput, topk: int = 5):
    x_np   = build_x(data.my_picks, data.opp_picks)
    mask_np = np.zeros(HERO_COUNT, dtype=np.bool_)
    mask_np[data.my_picks + data.opp_picks] = True

    with torch.no_grad(), torch.cuda.amp.autocast():
        x    = torch.from_numpy(x_np).unsqueeze(0)
        mask = torch.from_numpy(mask_np).unsqueeze(0)
        logits = model(x, mask)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    top_ids = probs.argsort()[-topk:][::-1]
    return {"recommendations": [
        {"hero_id": int(h), "score": round(float(probs[h]), 4)} for h in top_ids
    ]}
