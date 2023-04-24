from . import BaseDataset, register_dataset
from typing import Dict, List, Any
import bisect
import pickle
import os
import torch

from transformers import AutoTokenizer, DataCollatorWithPadding

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

        self.mimiciii_path = os.path.join(self.data_path, "preprocessed_mimiciii.pickle")
        self.mimiciv_path  = os.path.join(self.data_path, "preprocessed_mimiciv.pickle")
        self.eicu_path     = os.path.join(self.data_path, "preprocessed_eicu.pickle")

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

        # MIMIC-III
        # dict_keys(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'STARTDATE', 'ENDDATE', 'DRUG_TYPE', 'DRUG', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC', 'FORMULARY_DRUG_CD', 'GSN', 'NDC', 'PROD_STRENGTH', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'ROUTE'])
        # dict_keys(['SUBJECT_ID', 'HADM_ID', 'ITEMID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'VALUEUOM', 'FLAG', 'ICUSTAY_ID'])
        # dict_keys(['SUBJECT_ID', 'HADM_ID', 'ICUSTAY_ID', 'CHARTTIME', 'ITEMID', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM', 'STORETIME', 'CGID', 'ORDERID', 'LINKORDERID', 'STOPPED', 'NEWBOTTLE', 'ORIGINALAMOUNT', 'ORIGINALAMOUNTUOM', 'ORIGINALROUTE', 'ORIGINALRATE', 'ORIGINALRATEUOM', 'ORIGINALSITE', 'ENDTIME', 'ORDERCATEGORYNAME', 'SECONDARYORDERCATEGORYNAME', 'ORDERCOMPONENTTYPEDESCRIPTION', 'ORDERCATEGORYDESCRIPTION', 'PATIENTWEIGHT', 'TOTALAMOUNT', 'TOTALAMOUNTUOM', 'ISOPENBAG', 'CONTINUEINNEXTDEPT', 'CANCELREASON', 'STATUSDESCRIPTION', 'COMMENTS_EDITEDBY', 'COMMENTS_CANCELEDBY', 'COMMENTS_DATE'])

        # MIMIC-IV
        # dict_keys(['subject_id', 'hadm_id', 'pharmacy_id', 'poe_id', 'poe_seq', 'starttime', 'stoptime', 'drug_type', 'drug', 'formulary_drug_cd', 'gsn', 'ndc', 'prod_strength', 'form_rx', 'dose_val_rx', 'dose_unit_rx', 'form_val_disp', 'form_unit_disp', 'doses_per_24_hrs', 'route', 'stay_id'])
        # dict_keys(['subject_id', 'hadm_id', 'stay_id', 'starttime', 'endtime', 'storetime', 'itemid', 'amount', 'amountuom', 'rate', 'rateuom', 'orderid', 'linkorderid', 'ordercategoryname', 'secondaryordercategoryname', 'ordercomponenttypedescription', 'ordercategorydescription', 'patientweight', 'totalamount', 'totalamountuom', 'isopenbag', 'continueinnextdept', 'statusdescription', 'originalamount', 'originalrate'])
        # dict_keys(['labevent_id', 'subject_id', 'hadm_id', 'specimen_id', 'itemid', 'charttime', 'storetime', 'value', 'valuenum', 'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag', 'priority', 'comments', 'stay_id'])

        # EICU
        # dict_keys(['labid', 'patientunitstayid', 'labresultoffset', 'labtypeid', 'labname', 'labresult', 'labresulttext', 'labresultrevisedoffset', 'labmeasurename'])
        # dict_keys(['medicationid', 'patientunitstayid', 'drugorderoffset', 'drugstartoffset', 'drugivadmixture', 'drugordercancelled', 'drugname', 'drughiclseqno', 'dosage', 'routeadmin', 'frequency', 'loadingdose', 'prn', 'drugstopoffset', 'gtc'])
        # dict_keys(['infusiondrugid', 'patientunitstayid', 'infusionoffset', 'drugname', 'drugrate', 'infusionrate', 'drugamount', 'volumeoffluid', 'patientweight'])
        
        self.labs_formats = {
            "mimiciii": "{ITEMID}: {VALUENUM} {VALUEUOM} ({FLAG})",
            "mimiciv": "{itemid}: {valuenum} {valueuom} ({flag})",
            "eicu": "{labname}: {labresult} {labmeasurename} ({labresultoffset})",
        }
        self.prescrips_formats = {
            "mimiciii": "{DRUG_TYPE} - {DRUG} ({PROD_STRENGTH}): {DOSE_VAL_RX} {DOSE_UNIT_RX} ({ROUTE})",
            "mimiciv": "{drug_type} - {drug} ({prod_strength}): {dose_val_rx} {dose_unit_rx} ({route})",
            "eicu": "{drugname}: {drugstartoffset} ({routeadmin}, {drugordercancelled})",
        }
        self.inputs_formats = {
            "mimiciii": "{ITEMID}: {RATE} {RATEUOM} ({STOPPED})",
            "mimiciv": "{itemid}: {rate} {rateuom} ({stopped})",
            "eicu": "{drugname}: drugrate: {drugrate, ({infusionrate})",
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
            "input_ids": torch.stack(all_input_ids), # shape > (bs, 128 = max_sequence_length)
            "attention_mask": torch.stack(all_attention_mask),
            "labels": label,
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
        # input_ids, attention_mask, labels, intime, icustay_id
        origin_timesteps = [len(s['input_ids']) for s in samples]
        batch_max = max(origin_timesteps)

        input_ids = [torch.cat([s['input_ids'], torch.zeros(batch_max-len(s['input_ids']), self.MODEL_MAX_LENGTH, dtype=s['input_ids'].dtype)], dim=0) for s in samples]
        attention_masks = [torch.cat([s['attention_mask'], torch.zeros(batch_max-len(s['input_ids']), self.MODEL_MAX_LENGTH, dtype=s['input_ids'].dtype)], dim=0) for s in samples]
        intimes = [s['intime'] for s in samples]
        labels = [s['labels'] for s in samples]
        icustay_ids = [s['icustay_id'] for s in samples]
        
        return {
            "input_ids": input_ids, 
            "attention_mask": attention_masks,
            "labels": labels,
            "intime": intimes,
            "icustay_id": icustay_ids,
            "timesteps": origin_timesteps,
        }
