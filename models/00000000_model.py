from typing import Dict

from . import BaseModel, register_model

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel

@register_model("00000000_model")
class MyModel00000000(BaseModel):
    """
    TODO:
        create your own model here to handle heterogeneous EHR formats.
        Rename the class name and the file name with your student number.
    
    Example:
    - 20218078_model.py
        @register_model("20218078_model")
        class MyModel20218078(BaseModel):
            (...)
    """
    
    def __init__(
        self,
        # ...,
        **kwargs,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(kwargs['model_path'], return_dict=False)
        # Freeze bert layers
        if not kwargs['bert_unfreeze']:
            for p in self.bert.parameters():
                p.requires_grad = False

        # Event Aggregator
        self.agg = RNNModel()

        # Custom layers
        custom_hidden_dim = 64
        self.customs = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.agg.final_dim, custom_hidden_dim), # (128, 64)
            nn.Dropout(0.25),
            nn.Linear(custom_hidden_dim, 52)
        )

    def get_logits(cls, net_output):
        """get logits from the net's output.
        
        Note:
            Assure that get_logits(...) should return the logits in the shape of (batch, 52)
        """

        # logits = torch.zeros_like(net_output, device=net_output.device)

        # sig = nn.Sigmoid()
        # softmax = nn.Softmax(dim=1)

        # # binary
        # idx = 22
        # logits[:, :idx] = sig(net_output[:, :idx])

        # # multiclass
        # class_num = [6, 6, 5, 5, 5, 3]
        # for n in class_num:
        #     logits[:, idx:idx+n] = softmax(net_output[:, idx:idx+n])
        #     idx += n
            
        return net_output
    
    def get_targets(self, sample):
        """get targets from the sample
        
        Note:
            Assure that get_targets(...) should return the ground truth labels
                in the shape of (batch, 28)
        """
        # string to list
        targets = [l.strip('][').split(', ') for l in sample['labels']]
        # string element to list
        targets = [list(map(eval, t)) for t in targets]

        return torch.tensor(targets).to(sample['input_ids'].device)

    def forward(
        self,
        input_ids, attention_mask, labels,
        **kwargs
    ):
        """
        Note:
            the key should be corresponded with the output dictionary of the dataset you implemented.
        
        Example:
            class MyDataset(...):
                ...
                def __getitem__(self, index):
                    (...)
                    return {"data_key": data, "label": label}
            
            class MyModel(...):
                ...
                def forward(self, data_key, **kwargs):
                    (...)
        """
        _, embedding = self.bert(
            input_ids=input_ids.squeeze(0),
            attention_mask=attention_mask.squeeze(0) # 원래 기대하는 shape는 (bs, max sequence length)
        ) # output > (30, 768)

        embedding = torch.flip(embedding, [0]) # reverse order
        agg = self.agg(embedding, 128)  # output > (1, 128)
        output = self.customs(agg) # (1, 52)

        return output

# RNN for Aggregation    
class RNNModel(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self.batch_size = 1
        input_size = 768
        self.hidden_dim = 256
        self.final_dim = 128
        drop_out = 0.3
        self.n_layers = 3

        self.model = nn.GRU(
            input_size,
            self.hidden_dim,
            dropout=drop_out,
            batch_first=True,
            bidirectional=False,
            num_layers=self.n_layers
        )

        self.dropout = nn.Dropout(0.3)

        self.final_proj = nn.Linear(
            self.hidden_dim,
            self.final_dim
        )

    def forward(self, x, seq_len, **kwargs):
        # self.model.flatten_parameters()
        
        # unsqueeze 
        x = x.unsqueeze(0)

        h_0 = self._init_state(batch_size=x.size(0))
        x, _ = self.model(x, h_0)
        h_t = x[:, -1]

        self.dropout(h_t)
        output = self.final_proj(h_t)
        
        return output
    
    def _init_state(self, batch_size=1):
        weight = next(self.parameters()).data
        weight = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
        return weight

    # def pack_pad_seq(self, x, lengths):
    #     lengths = lengths.squeeze(-1).cpu()

    #     packed =  pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    #     output, _ = self.model(packed)
    #     output_seq, output_len = pad_packed_sequence(output, batch_first=True, padding_value=0)
        
    #     return output_seq, output_len
