from . import BaseDataset, register_dataset
from typing import Dict, List, Any
import bisect
import pickle
import os
import torch

from transformers import AutoTokenizer

@register_dataset("00000000_dataset")
class MyDataset00000000(BaseDataset):
    """
    TODO:
        create your own dataset here.
        Rename the class name and the file name with your student number
    
    Example:
    - 20218078_dataset.py
        @register_dataset("20218078_dataset")
        class MyDataset20218078(BaseDataset):
            (...)
    """

    # PRETRAINED_MODEL_NAME_OR_PATH = "emilyalsentzer/Bio_ClinicalBERT"
    MODEL_MAX_LENGTH = 128

    @staticmethod
    def cumsum(sequences):
        r, s = [], 0
        for e in sequences:
            l = len(e)
            r.append(l + s)
            s += l
        return r


    def __init__(
        self,
        data_path: str, # data_path should be a path to the processed features
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path

        self.mimiciii_path = os.path.join(self.data_path, "mimiciii.pickle")
        self.mimiciv_path  = os.path.join(self.data_path, "mimiciv.pickle")
        self.eicu_path     = os.path.join(self.data_path, "eicu.pickle")

        self.mimiciii = pickle.load(open(self.mimiciii_path, "rb")) if os.path.exists(self.mimiciii_path) else []
        self.mimiciv  = pickle.load(open(self.mimiciv_path, "rb")) if os.path.exists(self.mimiciv_path) else []
        self.eicu     = pickle.load(open(self.eicu_path, "rb")) if os.path.exists(self.eicu_path) else []   
        
        self.raw_datasets = [self.mimiciii, self.mimiciv, self.eicu]
        self.cumulative_sizes = self.cumsum(self.raw_datasets)

        self.tokenizer = AutoTokenizer.from_pretrained(kwargs['model_path'])

        self.bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.cls_token_id
        self.eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.sep_token_id
        self.sep_token_id = self.tokenizer.sep_token_id
        self.pad_token_id = self.tokenizer.pad_token_id

        self.labs_formats = {
            "mimiciii": "{ITEMID}: {VALUE} {VALUEUOM}",
            "mimiciv": "{itemid}: {value} {valueuom}",
            "eicu": "{labname}: {labresult} {labmeasurename}",
        }
        self.prescrips_formats = {
            "mimiciii": "{DRUG_TYPE} - {DRUG} ({PROD_STRENGTH}): {DOSE_VAL_RX} {DOSE_UNIT_RX}",
            "mimiciv": "{drug_type} - {drug} ({prod_strength}): {dose_val_rx} {dose_unit_rx}",
            "eicu": "{drugname}: {dosage} (frequency: {frequency})",
        }
        self.inputs_formats = {
            "mimiciii": "{ITEMID}: {AMOUNT} {AMOUNTUOM}",
            "mimiciv": "{itemid}: {amount} {amountuom} (rate: {rate} {rateuom})",
            "eicu": "{drugname} - drugrate: {drugrate} infusionrate: {infusionrate} drugamount: {drugamount} volumeoffluid: {volumeoffluid}",
        }

    
    def __getitem__(self, index):
        """
        Note:
            You must return a dictionary here or in collator so that the data loader iterator
            yields samples in the form of python dictionary. For the model inputs, the key should
            match with the argument of the model's forward() method.
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
        if index < 0:
            if -index > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            index = len(self) + index
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, index)
        if dataset_idx == 0:
            sample_idx = index
        else:
            sample_idx = index - self.cumulative_sizes[dataset_idx - 1]
        
        if dataset_idx == 0:
            dataset_name = "mimiciii"
        elif dataset_idx == 1:
            dataset_name = "mimiciv"
        else:
            dataset_name = "eicu"

        return self.preprocess(self.raw_datasets[dataset_idx][sample_idx], dataset_name)
    

    def tokenize(self, items: List[Dict[str, Any]], format_str: str):
        all_input_ids = [self.bos_token_id]
        all_attention_mask = [1]

        for item in items:
            input_str = format_str.format(**item)
            tokenized_inputs = self.tokenizer.encode(input_str, add_special_tokens=False)
            all_input_ids.extend(tokenized_inputs)
            all_attention_mask.extend([1] * len(tokenized_inputs))

            all_input_ids.append(self.sep_token_id)
            all_attention_mask.append(1)
        
        all_input_ids.append(self.eos_token_id)
        all_attention_mask.append(1)

        # PAD or TRUNCATE
        if len(all_input_ids) > self.MODEL_MAX_LENGTH:
            all_input_ids = all_input_ids[:self.MODEL_MAX_LENGTH-1]
            all_input_ids.append(self.eos_token_id)
            all_attention_mask = all_attention_mask[:self.MODEL_MAX_LENGTH]
            
        else:
            all_input_ids.extend([self.pad_token_id] * (self.MODEL_MAX_LENGTH - len(all_input_ids)))
            all_attention_mask.extend([0] * (self.MODEL_MAX_LENGTH - len(all_attention_mask)))

        input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)

        return input_ids, attention_mask
        

    def preprocess(self, sample: Dict[str, Any], dataset_name: str) -> Dict[str, torch.Tensor]:
        """
        Note:
            You can implement this method to preprocess the sample before returning it.
            This method is called in __getitem__ method.
        """
        icustay_id = sample["icustay_id"] if "mimic" in dataset_name else sample["patientunitstayid"]
        label = sample["label"]
        intime = sample["intime"] if "mimic" in dataset_name else ""
        
        events: List[str, Any] = sample["data"]

        all_input_ids = []
        all_attention_mask = []

        if dataset_name in ["mimiciii", "mimiciv"]:
            for event in events:
                if "time" in event:
                    time = event["time"]
                else:
                    time = None

                # Padding is done in the tokenize function
                # Therefore, all the input_ids and attention_mask should have the same length
                if "labs" in event and len(event["labs"]) > 0:
                    input_ids, attention_mask = self.tokenize(event["labs"], self.labs_formats[dataset_name])
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)

                if "prescrips" in event and len(event["prescrips"]) > 0:
                    input_ids, attention_mask = self.tokenize(event["prescrips"], self.prescrips_formats[dataset_name])
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
                
                if "inputs" in event and len(event["inputs"]) > 0:
                    input_ids, attention_mask = self.tokenize(event["inputs"], self.inputs_formats[dataset_name])
                    all_input_ids.append(input_ids)
                    all_attention_mask.append(attention_mask)
        
        elif dataset_name == "eicu":

            if "labs" in events and len(events["labs"]) > 0:
                input_ids, attention_mask = self.tokenize(events["labs"], self.labs_formats[dataset_name])
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)

            if "prescrips" in events and len(events["prescrips"]) > 0:
                input_ids, attention_mask = self.tokenize(events["prescrips"], self.prescrips_formats[dataset_name])
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
            
            if "inputs" in events and len(events["inputs"]) > 0:
                input_ids, attention_mask = self.tokenize(events["inputs"], self.inputs_formats[dataset_name])
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)

        return {
            "input_ids": torch.stack(all_input_ids),
            "attention_mask": torch.stack(all_attention_mask),
            "label": label,
            "intime": intime,
            "icustay_id": icustay_id,
        }

    
    def __len__(self):
        return self.cumulative_sizes[-1]

    def collator(self, samples):
        """Merge a list of samples to form a mini-batch.
        
        Args:
            samples (List[dict]): samples to collate
        
        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        
        Note:
            You can use it to make your batch on your own such as outputting padding mask together.
            Otherwise, you don't need to implement this method.
        """

        raise NotImplementedError