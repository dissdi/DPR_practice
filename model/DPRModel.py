from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class DPRModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.p_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def load_checkpoint(self, epoch_dir):
        self.q_encoder = BertModel.from_pretrained(f"{epoch_dir}/q_encoder")
        self.p_encoder = BertModel.from_pretrained(f"{epoch_dir}/p_encoder")
        self.tokenizer = BertTokenizer.from_pretrained(f"{epoch_dir}/tokenizer")
        
    def tokenize(self, text_a, text_b, MAX_LENGTH):
        if text_b == None:
            tokens = self.tokenizer(
                text_a,
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True,
                return_tensors="pt"
            )
        else:
            tokens = self.tokenizer(
                text_a, text_b,
                padding="max_length",
                max_length=MAX_LENGTH,
                truncation=True,
                return_tensors="pt"
            )
        return tokens
        
    def encode_questions(self, q_input):
        return self.q_encoder(**q_input).last_hidden_state[:, 0, :]

    def encode_passages(self, p_input):
        out =  self.p_encoder(**p_input, output_hidden_states=True)
        cls0 = out.last_hidden_state[:, 0, :]
        cls_pooled = torch.zeros_like(cls0)
        for h in out.hidden_states[1:]:
            cls_pooled += h[:, 0, :]
        return cls_pooled/len(out.hidden_states[1:])
        
    def forward(self, q_emb, p_emb):
        return q_emb @ p_emb.T