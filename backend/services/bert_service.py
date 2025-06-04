import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn

_MODEL_DIR = r"G:\AI20799\backend\services\saved_model_time_series"

class BertWithTimeFeatures(nn.Module):
    def __init__(self, bert_model_name='hfl/chinese-roberta-wwm-ext',
                 time_feat_dim=9, num_labels=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(hidden_size + time_feat_dim, num_labels)

    def forward(self, input_ids, attention_mask, time_features, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        combined = torch.cat([cls_output, time_features], dim=1)
        logits = self.classifier(combined)
        return logits

tokenizer = BertTokenizer.from_pretrained(_MODEL_DIR, local_files_only=True)
state_dict = torch.load(os.path.join(_MODEL_DIR, "full_model_state.pt"), map_location="cpu")

time_dim = state_dict["classifier.weight"].shape[1] - state_dict["bert.pooler.dense.weight"].shape[0]

model = BertWithTimeFeatures(time_feat_dim=time_dim)
model.load_state_dict(state_dict, strict=False)
model.eval()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

def build_time_features(ts: int) -> np.ndarray:
    from datetime import datetime
    dt = datetime.fromtimestamp(ts) if ts else datetime.now()
    hour_sin = np.sin(2 * np.pi * dt.hour / 24)
    hour_cos = np.cos(2 * np.pi * dt.hour / 24)
    feats = np.zeros(9, dtype=np.float32)
    feats[0:2] = [hour_sin, hour_cos]
    return feats

@torch.inference_mode()
def bert_predict(text: str, ts: int | None = None) -> dict:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=128, padding='max_length')
    time_feats = torch.tensor(build_time_features(ts)).unsqueeze(0)

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    time_feats = time_feats.to(DEVICE)

    logits = model(**inputs, time_features=time_feats)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    label = int(np.argmax(probs))
    return {"label": label, "prob": float(probs[label]), "logits": probs.tolist()}
