import functools
import pathlib
import time
import typing
import warnings

import numpy as np
import openai
from openai import OpenAI

client = OpenAI()
import torch

from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from probsem.abstract import Object, IModel
from probsem.utils import tokenize, detokenize

# TODO: The 'openai.api_key_path' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_key_path=str(pathlib.Path.home() / ".openai_api_key"))'
# openai.api_key_path = str(pathlib.Path.home() / ".openai_api_key")

class Model(Object):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._id = model_id
        # self._model: IModel
        setattr(self, "_model", OpenAIModel(self._id))
        
        # # New way to list engines using the OpenAI API 1.0.0
        # try:
        #     # TODO: The resource 'Engine' has been deprecated
        #     # openai_engines = openai.Engine.list()
        #     engine_ids = [engine.id for engine in openai_engines.data]
        #     if self._id in engine_ids:
        #         self.info("Model ID found in OpenAI engines.")
        #         setattr(self, "_model", OpenAIModel(self._id))
        #     else:
        #         self.info("Model ID not found in OpenAI engines. Checking HuggingFace.")
        #         setattr(self, "_model", HuggingFaceModel(self._id))
        # except Exception as e:
        #     # Handle exceptions appropriately
        #     self.warn(f"Error accessing OpenAI engines: {str(e)}")

    def score(self, full_text: str, eval_text: str) -> np.float64:
        return self._model.score(full_text, eval_text)

# class Model(Object):
#     def __init__(self, model_id: str) -> None:
#         super().__init__()
#         self._id = model_id
#         self._model: IModel
#         openai_engines = [engine["id"] for engine in openai.Engine.list()["data"]]
#         if self._id in openai_engines:

#             self.info("Model ID found in OpenAI engines.")
#             setattr(self, "_model", OpenAIModel(self._id))
#         else:
#             print("selfid",self._id )
#             self.info("Model ID not found in OpenAI engines. Checking HuggingFace." )
#             setattr(self, "_model", HuggingFaceModel(self._id))

#     def score(
#         self,
#         full_text: str,
#         eval_text: str,
#         normalize: bool = True,
#         temperature: float = 1.0,
#     ) -> np.float64:
#         logp, num_eval = self._model.score(full_text, eval_text)
#         if normalize:
#             logp /= np.sqrt(num_eval)
#         return logp / temperature

class OpenAIModel(Object):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._id = model_id
        self.info(f"Selected OpenAI {self._id} model.")
        self.client = openai.OpenAI()
    # def _get_response(self, text: str, retry_after=10):
    #     try:
    #         print("text", text)
    #         response = client.completions.create(model=self._id,
    #         prompt=text,
    #         max_tokens=0,  # Adjust max_tokens as needed
    #         n=1,
    #         logprobs=0,
    #         echo=True)
    #         return response
    #     except openai.RateLimitError:  # Correctly reference the RateLimitError
    #         self.warn(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
    #         time.sleep(retry_after)
    #         return self._get_response(text, retry_after * 2)
        
    def score(self, full_text: str, eval_text: str) -> np.float64:
        response = self._get_response(full_text)
        return self._calculate_score_from_response(response, eval_text)

    def _calculate_score_from_response(self, response, eval_text):
        # Logic to analyze the response and calculate a score
        # This is a placeholder implementation and should be adapted based on your requirements
        # For example, checking if the response text contains certain keywords or phrases
        # that indicate a preference for one of the options.
        first_choice = response.choices[0]  # Access the first Choice object
        message = first_choice.message      # Access the ChatCompletionMessage object
        content = message.content 

        print('content',content)
        

        # Example scoring logic
        if eval_text in content:
            return 1.0  # Maximum score if the response contains the evaluated text
        else:
            return 0.0  # Minimum score otherwise
    def _get_response(self, text: str, retry_after=10):
        # print('text', text)
        try:
            completion = self.client.chat.completions.create(
                model=self._id,
                messages=[
                    {"role": "system", "content": "You are an expert in Webppl code and an assistant to a probability lab: based on the given code, give me who you think will win in each scenario, and the probability that you think that person will win with (how confident you are). Only output the asnwer to the question in the format [winner, probability of winning] do not output anything else "},
                    {"role": "user", "content": text}
                ]
            )
            return completion
        except openai.RateLimitError:
            self.warn(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
            time.sleep(retry_after)
            return self._get_response(text, retry_after * 2)

# class OpenAIModel(Object, IModel):
#     def __init__(self, model_id: str) -> None:
#         super().__init__()
#         self._id = model_id
#         self.info(f"Selected OpenAI {self._id} model.")

#     def _get_response(
#         self, text: str, retry_after=10
#     ) -> openai.openai_object.OpenAIObject:
#         try:
#             return openai.ChatCompletion.create(
#                 engine=self._id,
#                 prompt=text,
#                 max_tokens=0,
#                 logprobs=0,
#                 echo=True,
#             )
#         except openai.error.RateLimitError:
#             self.warn(f"Rate limit exceeded. Retrying after {retry_after} seconds.")
#             time.sleep(retry_after)
#             return self._get_response(text, retry_after * 2)

#     def score(self, full_text: str, eval_text: str) -> typing.Tuple[np.float64, int]:
#         full_resp = self._get_response(full_text)
#         eval_resp = self._get_response(eval_text)
#         num_eval = eval_resp["usage"]["total_tokens"]
#         get_tokens = lambda resp: resp["choices"][0]["logprobs"]["tokens"]
#         assert get_tokens(full_resp)[-num_eval:] == get_tokens(eval_resp)
#         logp = np.sum(full_resp["choices"][0]["logprobs"]["token_logprobs"][-num_eval:])
#         return logp, num_eval


class HuggingFaceModel(Object):
    def __init__(self, model_id: str) -> None:
        super().__init__()
        self._id = model_id
        self.info(f"Attempting to load HuggingFace {self._id} model...")
        try:
            self._config = AutoConfig.from_pretrained(self._id)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._id, add_prefix_space=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self._id, torch_dtype=torch.float32, low_cpu_mem_usage=True
            )
            self._model.eval()
        except Exception as invalid_id:
            raise ValueError(
                "model must be valid HuggingFace CausalLM."
            ) from invalid_id
        self._set_torch_device()
        self.info(f"Successfully loaded pretrained {self._id} model on {self._device}.")

    def _set_torch_device(self) -> None:
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
            torch.set_default_tensor_type(torch.cuda.FloatTensor)  # type: ignore
            try:
                self._model = self._model.to(self._device)
                return
            except RuntimeError:
                self._device = torch.device("cpu")
                torch.set_default_tensor_type(torch.FloatTensor)
                self._model = self._model.to(self._device)
        else:
            self._device = torch.device("cpu")
            torch.set_default_tensor_type(torch.FloatTensor)
            self._model = self._model.to(self._device)

    @functools.lru_cache(maxsize=128)
    def _encode_text(self, text: str) -> typing.Dict[str, torch.Tensor]:
        return self._tokenizer(
            tokenize(text), is_split_into_words=True, return_tensors="pt"
        ).to(self._device)

    def _decode_text(self, tokens: torch.Tensor) -> str:
        return detokenize(self._tokenizer.decode(tokens, skip_special_tokens=True))

    def score(self, full_text: str, eval_text: str) -> typing.Tuple[np.float64, int]:
        with torch.no_grad():
            inputs = self._encode_text(full_text)
            num_eval = self._encode_text(eval_text)["input_ids"].shape[1]
            tokens = inputs["input_ids"]
            mask = inputs["attention_mask"]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                outputs = self._model(input_ids=tokens, attention_mask=mask)
            loss = torch.nn.CrossEntropyLoss(reduction="none")(
                outputs.logits[..., :-1, :]
                .contiguous()
                .view(-1, outputs.logits.size(-1)),
                tokens[..., 1:].contiguous().view(-1),
            ).view(tokens.size(0), tokens.size(-1) - 1)
            loss = loss * mask[..., 1:].contiguous()
            loss = loss[:, -num_eval:].sum(dim=1)
            logp = -loss.cpu().detach().item()
            return logp, num_eval
