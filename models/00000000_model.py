from typing import Dict

from . import BaseModel, register_model

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel

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
        
        self.bert = AutoModel.from_pretrained(kwargs['model_path'], return_dict=False)
        # Freeze all BERT layers
        if not kwargs.get('bert_unfreeze', False):
            for p in self.bert.parameters():
                p.requires_grad = False
    
        # Freeze all BERT layers except the last 2 encoder layers
        elif kwargs.get('bert_unfreeze_top3', False):
            for i, module in enumerate(self.bert.encoder.children()):
                if i < 8:
                    for param in module.parameters():
                        param.requires_grad = False
                        
                        
        self.embedding_dim = 768
        self.transformer_dim = 512
        
        
        # intermediate linear btw embedding model and Aggregator
        self.Inter_Linear = nn.Linear(self.embedding_dim, self.transformer_dim)
        
        
        # Event Aggregator
        self.agg = Transformer()

        # Custom layers
        self.custom_hidden_dim = 128
        self.customs = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.agg.final_dim, self.custom_hidden_dim), # (256, 128)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.custom_hidden_dim, 52)
        )

    def get_logits(cls, net_output):
        """get logits from the net's output.
        
        Note:
            Assure that get_logits(...) should return the logits in the shape of (batch, 52)
        """

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

        return torch.tensor(targets).to(sample['input_ids'][0].device)

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
        embeddings = [torch.mean(self.bert(input_ids=i, attention_mask=m)[0], dim=1) for i, m in zip(input_ids, attention_mask)] # average-pooling
        embeddings = self.Inter_Linear(torch.stack(embeddings)) # (bs, max_timestep, 768) -> (bs, max_timestep, 512)
        
        agg = self.agg(embeddings)  # (bs, max_timestep, 512) -> (bs, 256)
        output = self.customs(agg) # (bs, 52)

        return output

# Transformer for Aggregation    
class Transformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.dim_model = 512
        self.num_heads = 8
        self.num_encoder_layers = 6
        self.dropout_p = 0.2
        self.final_dim = 256
        
        self.encoder_layer = nn.TransformerEncoderLayer(self.dim_model, self.num_heads)
        self.encoder = nn.TransformerEncoder(
            encoder_layer = self.encoder_layer,
            num_layers = self.num_encoder_layers
        )

        self.final_proj = nn.Sequential(
            nn.Linear(
                self.dim_model,
                self.final_dim
            )
        )

    def forward(self, x):
        output = torch.mean(self.encoder(x),dim=1) # (bs, max_timestep, 512) -> (bs, 512)
        output = self.final_proj(output) # (bs, 512) -> (bs, 256)
        return output
    
    # def _init_state(self, batch_size=1):
    #     weight = next(self.parameters()).data
    #     weight = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
    #     return weight

    # def pack_pad_seq(self, x, lengths):
    #     lengths = torch.tensor(lengths).cpu()

    #     packed =  pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
    #     output, _ = self.model(packed)
    #     output_seq, output_len = pad_packed_sequence(output, batch_first=True, padding_value=0)
        
    #     return output_seq, output_len
