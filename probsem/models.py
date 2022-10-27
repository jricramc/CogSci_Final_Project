import typing

import numpy as np
import torch
from transformers import CodeGenConfig, CodeGenTokenizer, CodeGenForCausalLM

from probsem.abstract import Object


class Model(Object):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._id = model_id
        try:
            self._config = CodeGenConfig.from_pretrained(self._id)
            self._tokenizer = CodeGenTokenizer.from_pretrained(self._id)
            self._model = CodeGenForCausalLM.from_pretrained(self._id)
            self._model.eval()
        except Exception as invalid_id:
            raise ValueError(
                "model must be valid HuggingFace CodeGenCausalLM."
            ) from invalid_id
        self._set_torch_device()
        self.info(f"Loaded pretrained {self._id} model on {self._device}.")

    def _set_torch_device(self) -> None:
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
            try:
                self._model = self._model.to(self._device)
                return
            except RuntimeError:
                self._device = torch.device("cpu")
                torch.set_default_tensor_type(torch.FloatTensor)  # type: ignore
                self._model = self._model.to(self._device)
        else:
            self._device = torch.device("cpu")
            self._model = self._model.to(self._device)

    def _tokenize(self, text: str) -> typing.Dict[str, torch.Tensor]:
        return self._tokenizer(text, return_tensors="pt").to(self._device)

    def score(
        self, full_text: str, eval_text: str, normalize: bool = False
    ) -> np.float64:
        with torch.no_grad():
            inputs = self._tokenize(full_text)
            n_eval = self._tokenize(eval_text)["input_ids"].shape[1]
            tokens = inputs["input_ids"]
            outputs = self._model(**inputs, labels=tokens)
            loss = torch.nn.CrossEntropyLoss(reduction="none")(
                outputs.logits[..., :-1, :]
                .contiguous()
                .view(-1, outputs.logits.size(-1)),
                tokens[..., 1:].contiguous().view(-1),
            ).view(tokens.size(0), tokens.size(-1) - 1)
            loss = loss * inputs["attention_mask"][..., 1:].contiguous()
            loss = loss[:, -n_eval:].sum(dim=1)
            if normalize:
                loss -= np.log(n_eval)
            logp = -loss.cpu().detach().item()
        return logp
