from transformers import AutoModel, AutoTokenizer
from utils import truncate_sentences, pool_and_normalize, set_device
import torch
from abc import ABC, abstractmethod
from utils import PAD_TOKEN, SEPARATOR_TOKEN, CLS_TOKEN, MASK_TOKEN

# https://github.com/bigcode-project/bigcode-encoder/tree/master


def prepare_tokenizer(tokenizer_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=True)

    tokenizer.add_special_tokens({"pad_token": PAD_TOKEN})
    tokenizer.add_special_tokens({"sep_token": SEPARATOR_TOKEN})
    tokenizer.add_special_tokens({"cls_token": CLS_TOKEN})
    tokenizer.add_special_tokens({"mask_token": MASK_TOKEN})
    return tokenizer


class BaseEncoder(torch.nn.Module, ABC):

    def __init__(self, device, max_input_len, maximum_token_len, model_name):
        super().__init__()

        self.model_name = model_name
        self.tokenizer = prepare_tokenizer(model_name)
        self.encoder = (
            AutoModel.from_pretrained(model_name, token=True).to(device).eval()
        )
        self.device = device
        self.max_input_len = max_input_len
        self.maximum_token_len = maximum_token_len

    @abstractmethod
    def forward(
        self,
    ):
        pass

    def encode(self, input_sentences, batch_size=32, **kwargs):

        truncated_input_sentences = truncate_sentences(
            input_sentences, self.max_input_len
        )

        n_batches = len(truncated_input_sentences) // batch_size + int(
            len(truncated_input_sentences) % batch_size > 0
        )

        embedding_batch_list = []

        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(truncated_input_sentences))

            with torch.no_grad():
                embedding_batch_list.append(
                    self.forward(truncated_input_sentences[start_idx:end_idx])
                    .detach()
                    .cpu()
                )

        input_sentences_embedding = torch.cat(embedding_batch_list)

        return [emb.squeeze().numpy() for emb in input_sentences_embedding]


class StarEncoder(BaseEncoder):

    def __init__(self, device, max_input_len, maximum_token_len):
        super().__init__(
            device, max_input_len, maximum_token_len, model_name="bigcode/starencoder"
        )

    def forward(self, input_sentences):

        inputs = self.tokenizer(
            [f"{CLS_TOKEN}{sentence}{SEPARATOR_TOKEN}" for sentence in input_sentences],
            padding="longest",
            max_length=self.maximum_token_len,
            truncation=True,
            return_tensors="pt",
        )

        outputs = self.encoder(**set_device(inputs, self.device))
        embedding = pool_and_normalize(outputs.hidden_states[-1], inputs.attention_mask)

        return embedding


class CodeBERT(BaseEncoder):
    def __init__(self, device, max_input_len, maximum_token_len):
        super().__init__(
            device,
            max_input_len,
            maximum_token_len,
            model_name="microsoft/codebert-base",
        )

        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

    def forward(self, input_sentences):
        inputs = self.tokenizer(
            [sentence for sentence in input_sentences],
            padding="longest",
            max_length=self.maximum_token_len,
            truncation=True,
            return_tensors="pt",
        )

        inputs = set_device(inputs, self.device)

        outputs = self.encoder(inputs["input_ids"], inputs["attention_mask"])

        embedding = outputs["pooler_output"]

        return torch.cat(
            [
                torch.nn.functional.normalize(torch.Tensor(el)[None, :])
                for el in embedding
            ]
        )
