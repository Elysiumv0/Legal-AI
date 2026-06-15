import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "BAAI/bge-m3"
MAX_LENGTH = 512


class EmbeddingModel:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_length: int = MAX_LENGTH,
        device: str = "auto",
    ):
        print(f"Loading embedding model: {model_name}")
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        outputs = self.model(**encoded)
        embeddings = self._pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().float().numpy()

    def _pool(self, hidden_states, attention_mask):
        return hidden_states[:, 0]

    def unload(self):
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
