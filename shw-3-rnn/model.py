import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super(LanguageModel, self).__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length

        """
        YOUR CODE HERE (⊃｡•́‿•̀｡)⊃━✿✿✿✿✿✿
        Create necessary layers
        """
        self.embedding = nn.Embedding(self.vocab_size, embed_size)
        self.rnn = rnn_type(embed_size, hidden_size, num_layers=rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(indices)
        packed_embeds = nn.utils.rnn.pack_padded_sequence(embeds, lengths.cpu(), 
                                                          batch_first=True, enforce_sorted=False)
        packed_output, _ = self.rnn(packed_embeds)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        logits = self.linear(output)
        return logits



    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        device = next(self.parameters()).device
        #Кодируем префикс; не забываем добавить токен BOS (начало последовательности)
        if prefix:
            prefix_ids = self.dataset.text2ids(prefix)
            # Ожидается, что text2ids для строки вернёт список int
            if not isinstance(prefix_ids, list) or not all(isinstance(x, int) for x in prefix_ids):
                raise ValueError("Ожидается, что text2ids вернёт список int для строки-префикса.")
        else:
            prefix_ids = []

        # Начинаем последовательность с BOS токена
        tokens = [self.dataset.bos_id] + prefix_ids
        hidden = None

        # Пропускаем через RNN все токены префикса для накопления состояния
        for token in tokens:
            input_tensor = torch.tensor([[token]], dtype=torch.long, device=device)  # форма: (1,1)
            embed = self.embedding(input_tensor)  # (1,1,embed_size)
            _, hidden = self.rnn(embed, hidden)

        generated_tokens = tokens[:]
        while len(generated_tokens) < self.max_length:
            input_tensor = torch.tensor([[generated_tokens[-1]]], dtype=torch.long, device=device)
            embed = self.embedding(input_tensor)
            output, hidden = self.rnn(embed, hidden)  # output: (1,1,hidden_size)
            logits = self.linear(output.squeeze(1))  # (1, vocab_size)
            logits = logits / temp  # масштабирование по температуре
            probs = torch.softmax(logits, dim=-1)  # (1, vocab_size)
            next_token = torch.multinomial(probs, num_samples=1).item()  # выбор следующего токена
            generated_tokens.append(next_token)
            if next_token == self.dataset.eos_id:
                break

        # Убираем BOS токен, а также токены после EOS (если он сгенерировался)
        generated_tokens = generated_tokens[1:]
        if self.dataset.eos_id in generated_tokens:
            eos_index = generated_tokens.index(self.dataset.eos_id)
            generated_tokens = generated_tokens[:eos_index]

        generated = self.dataset.ids2text(generated_tokens)
        return generated