from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class DPR_MultiCarry(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.p_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.cls_indices = []

    def load_checkpoint(self, epoch_dir):
        self.q_encoder = BertModel.from_pretrained(f"{epoch_dir}/q_encoder")
        self.p_encoder = BertModel.from_pretrained(f"{epoch_dir}/p_encoder")
        self.tokenizer = BertTokenizer.from_pretrained(f"{epoch_dir}/tokenizer")
        
    def tokenize(self, text_a, text_b, MAX_LENGTH):
        if text_b == None:
            tokens = self.tokenizer(
                text_a,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            )
        else:
            tokens = self.tokenizer(
                text_a, text_b,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            )
        input_ids = tokens["input_ids"].tolist()
        attention_mask = tokens["attention_mask"].tolist()
        indexes = [102, 77, 52, 26, 0]
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id
        for idx in indexes:
            input_ids[0].insert(idx, cls_id)
            attention_mask[0].insert(idx, 1)
        for i, val in enumerate(input_ids[0]):
            if val == pad_id:
                input_ids[0].insert(i, sep_id)
                attention_mask[0].insert(i, 1)
                self.cls_indices.append(i)
                for _ in len(self.cls_indices):
                    if _ == 0:
                        continue
                    self.cls_indices[_] += 1
                break
        input_ids = torch.tensor(input_ids[0][:128], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask[0][:128], dtype=torch.long).unsqueeze(0)
        tokens["input_ids"] = input_ids
        tokens["attention_mask"] = attention_mask
        cls_id = self.tokenizer.cls_token_id
        cls_positions = (input_ids[0] == cls_id).nonzero(as_tuple=True)[0].tolist()
        return tokens, cls_positions
        
    def encode_questions(self, q_input, cls_positions):
        out =  self.q_encoder(**q_input, output_hidden_states=True)
        cls0 = out.last_hidden_state[:, 0, :]
        cls_pooled = torch.zeros_like(cls0)
        for h in out.hidden_states[1:]:
            for c in self.cls_indices:
                cls_pooled += h[:, 0, :]
        return cls_pooled/len(out.hidden_states[1:])

    def encode_passages(self, p_input, cls_positions):
        out =  self.p_encoder(**p_input, output_hidden_states=True)
        cls0 = out.last_hidden_state[:, 0, :]
        cls_pooled = torch.zeros_like(cls0)
        for h in out.hidden_states[1:]:
            cls_pooled += h[:, 0, :]
        return cls_pooled/len(out.hidden_states[1:])
        
    def forward(self, q_emb, p_emb):
        return q_emb @ p_emb.T