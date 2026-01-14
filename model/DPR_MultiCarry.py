from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch

class DPR_MultiCarry(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.p_encoder = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def load_checkpoint(self, epoch_dir):
        self.q_encoder = BertModel.from_pretrained(f"{epoch_dir}/q_encoder")
        self.p_encoder = BertModel.from_pretrained(f"{epoch_dir}/p_encoder")
        self.tokenizer = BertTokenizer.from_pretrained(f"{epoch_dir}/tokenizer")
        
    def tokenize(self, text_a, text_b, max_length):
        if text_b == None:
            tokens = self.tokenizer(
                text_a,
                max_length=max_length-6,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            )
        else:
            text = f"{text_a} {text_b}"
            tokens = self.tokenizer(
                text,
                max_length=max_length-6,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
                add_special_tokens=False
            )
        input_ids = tokens["input_ids"].tolist()
        attention_mask = tokens["attention_mask"].tolist()
        
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        # CLS token insert
        indexes = [4*max_length//5+3, 3*max_length//5+2, 2*max_length//5+1, max_length//5, 0]
        pad_find = False
        for idx in indexes:
            input_ids[0].insert(idx, cls_id)
            attention_mask[0].insert(idx, 1)
            
        # SEP before PAD
        for i, val in enumerate(input_ids[0]):
            if val == pad_id:
                input_ids[0].insert(i, sep_id)
                attention_mask[0].insert(i, 1)
                pad_find = True
                break
        if not pad_find:
            input_ids[0].append(sep_id)
            attention_mask[0].append(1)
                
        # truncate max length
        input_ids = torch.tensor(input_ids[0][:max_length], dtype=torch.long).unsqueeze(0)
        attention_mask = torch.tensor(attention_mask[0][:max_length], dtype=torch.long).unsqueeze(0)
        
        tokens["input_ids"] = input_ids
        tokens["attention_mask"] = attention_mask
        tokens.pop("token_type_ids", None)
        
        cls_id = self.tokenizer.cls_token_id
        cls_indices = (input_ids[0] == cls_id).nonzero(as_tuple=True)[0].tolist()
        return tokens, cls_indices
        
    def encode_questions(self, q_input, cls_indices):
        out =  self.q_encoder(**q_input, output_hidden_states=True)
        cls_pooled = torch.zeros_like(out.last_hidden_state[:, 0, :])
        for h in out.hidden_states[1:]:
            for c in cls_indices:
                cls_pooled += h[:, c, :]
        return cls_pooled/(len(out.hidden_states[1:])*max(1, len(cls_indices)))

    def encode_passages(self, p_input, cls_indices):
        out =  self.p_encoder(**p_input, output_hidden_states=True)
        cls_pooled = torch.zeros_like(out.last_hidden_state[:, 0, :])
        for h in out.hidden_states[1:]:
            for c in cls_indices:
                cls_pooled += h[:, c, :]
        return cls_pooled/(len(out.hidden_states[1:])*max(1, len(cls_indices)))
        
    def forward(self, q_emb, p_emb):
        return q_emb @ p_emb.T