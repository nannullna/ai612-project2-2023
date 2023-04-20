import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import pickle
import os

def get_parser():
    """
    Note:
        Do not add command-line arguments here when you submit the codes.
        Keep in mind that we will run your pre-processing code by this command:
        `python 00000000_preprocess.py ./train --dest ./output`
        which means that we might not be able to control the additional arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        metavar="DIR",
        default='./train/',
        help="root directory containing different ehr files to pre-process (usually, 'train/')"
    )
    parser.add_argument(
        "--dest",
        type=str,
        metavar="DIR",
        default= './outcome',
        help="output directory"
    )
    return parser


def make_icustay_id(III_lab, III_icustay, IV_lab, IV_prescrip, IV_icustay):
    ''''
    Make the ICUSTAY_ID columns for III_lab, IV_lab, IV_prescrip
    '''
    for i in tqdm(range(len(III_lab)), desc= '*** MAKE ICUSTAY ID (1/3) ***'): # i: lab index
        hadm_id = III_lab.HADM_ID.iloc[i]
        charttime = III_lab.CHARTTIME.iloc[i]
        idx = III_icustay[III_icustay.HADM_ID == hadm_id].index[0] # idx: icustay index
        intime = III_icustay.INTIME.iloc[idx]
        outtime = III_icustay.OUTTIME.iloc[idx]
        
        if datetime.strptime(intime,'%Y-%m-%d %H:%M:%S') <= datetime.strptime(charttime,'%Y-%m-%d %H:%M:%S') <= datetime.strptime(outtime,'%Y-%m-%d %H:%M:%S'):
            III_lab.ICUSTAY_ID.iloc[i] = III_icustay.ICUSTAY_ID.iloc[idx]

    for i in tqdm(range(len(IV_lab)), desc= '*** MAKE ICUSTAY ID (2/3)***'): # i: lab index
        hadm_id = IV_lab.hadm_id.iloc[i]
        charttime = IV_lab.charttime.iloc[i]
        idx = IV_icustay[IV_icustay.hadm_id == hadm_id].index[0] # idx: icustay index
        intime = IV_icustay.intime.iloc[idx]
        outtime = IV_icustay.outtime.iloc[idx]
        
        if datetime.strptime(intime,'%Y-%m-%d %H:%M:%S') <= datetime.strptime(charttime,'%Y-%m-%d %H:%M:%S') <= datetime.strptime(outtime,'%Y-%m-%d %H:%M:%S'):
            IV_lab.stay_id.iloc[i] = IV_icustay.stay_id.iloc[idx]

    for i in tqdm(range(len(IV_prescrip)), desc= '*** MAKE ICUSTAY ID (3/3)***'): # i: lab index
        hadm_id = IV_prescrip.hadm_id.iloc[i]
        starttime = IV_prescrip.starttime.iloc[i]
        idx = IV_icustay[IV_icustay.hadm_id == hadm_id].index[0] # idx: icustay index
        intime = IV_icustay.intime.iloc[idx]
        outtime = IV_icustay.outtime.iloc[idx]
        
        if datetime.strptime(intime,'%Y-%m-%d %H:%M:%S') <= datetime.strptime(starttime,'%Y-%m-%d %H:%M:%S') <= datetime.strptime(outtime,'%Y-%m-%d %H:%M:%S'):
            IV_prescrip.stay_id.iloc[i] = IV_icustay.stay_id.iloc[idx]

    return III_lab, IV_lab, IV_prescrip 


def look_up_table(data_type, target_df, dict_df, order):
    if data_type == 'III':
        itemid_col_name = 'ITEMID'
        #item_col_name = 'ITEM'
        label_col_name = 'LABEL'
    elif data_type == 'IV':
        itemid_col_name = 'itemid'
        #item_col_name = 'item'
        label_col_name = 'label'

    target_df[itemid_col_name] = target_df[itemid_col_name].astype(str)
    dict_df[itemid_col_name] = dict_df[itemid_col_name].astype(str)
    
    for i in tqdm(range(len(target_df)), desc= f'*** LOOK UP TABLE ({order}/4) ***'):
        itemid = str(target_df[itemid_col_name].iloc[i])
        #if i == 0:
        #    target_df[item_col_name] = ''
        target_df[itemid_col_name].iloc[i] = str(dict_df[dict_df[itemid_col_name]==itemid][label_col_name].iloc[0])
    
    # target_df.drop(itemid_col_name, axis=1, inplace=True)
    
    return target_df


def MIMIC_make_charttime_data(data_type, lab, prescrip, input_):
    '''
    Charttime data using lab, prescrip, input by id
    '''
    if data_type == 'III':
        LAB_TIME = 'CHARTTIME'
        PRESCRIP_TIME = 'STARTDATE'
        INPUT_TIME = 'CHARTTIME'
        
        # lab_columns = ['ITEMID', 'VALUE', 'VALUEUOM', 'FLAG']
        # prescrip_columns = ['DRUG_TYPE', 'DRUG', 'PROD_STRENGTH', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'FORM_VAL_DISP', 'FORM_UNIT_DISP', 'ROUTE']
        # input_columns = ['ITEMID', 'AMOUNT', 'AMOUNTUOM', 'ORIGINALAMOUNT']
    
    elif data_type == 'IV':
        LAB_TIME = 'charttime'
        PRESCRIP_TIME = 'starttime'
        INPUT_TIME = 'starttime'
        
        # lab_columns = ['itemid', 'value', 'valueuom', 'ref_range_lower', 'ref_range_upper', 'flag']
        # prescrip_columns = ['pharmacy_id', 'drug_type', 'drug', 'formulary_drug_cd', 'gsn', 'ndc', 'prod_strength', 'dose_val_rx', 'dose_unit_rx', 'form_val_disp', 'form_unit_disp', 'route']
        # input_columns = ['itemid', 'amount', 'amountuom', 'rate', 'rateuom', 'originalamount', 'originalrate']
    
    lab_columns = list(lab.columns)
    prescrip_columns = list(prescrip.columns)
    input_columns = list(input_.columns)
    
    data=[]
    unique_time_list = list(set(list(lab[LAB_TIME])+list(prescrip[PRESCRIP_TIME])+list(input_[INPUT_TIME])))
    unique_time_list.sort()
    for time in unique_time_list:
        by_charttime = {}
        labs=[]
        prescrips=[]
        inputs=[]

        lab_df = lab[lab[LAB_TIME]==time]
        prescrip_df = prescrip[prescrip[PRESCRIP_TIME]==time]
        input_df = input_[input_[INPUT_TIME]==time]

        for i in range(len(lab_df)):
            by_lab={}
            for col in lab_columns:
                by_lab[col] = lab_df[col].iloc[i]
            labs.append(by_lab)
            
        for i in range(len(prescrip_df)):
            by_prescrip={}
            for col in prescrip_columns:
                by_prescrip[col] = prescrip_df[col].iloc[i]
            prescrips.append(by_prescrip) 
            
        for i in range(len(input_df)):
            by_input={}
            for col in input_columns:
                by_input[col] = input_df[col].iloc[i]
            inputs.append(by_input)

        by_charttime['time'] = time
        by_charttime['labs'] = labs
        by_charttime['prescrips'] = prescrips
        by_charttime['inputs'] = inputs
        data.append(by_charttime)

    return data


def MIMIC_make_final_dict(data_type, lab_data, prescrip_data, input_data, label_data, icustay_data):
    '''
    Create final dictionary by collecting charttime data corresponding to each id
    '''
    if data_type == 'III':
        ICUSTAY_ID = 'ICUSTAY_ID'
        INTIME = 'INTIME'
    elif data_type == 'IV':
        ICUSTAY_ID = 'stay_id'
        INTIME = 'intime'
        
    final = []
    for id in tqdm(label_data[ICUSTAY_ID].unique(), desc= f'*** MAKE FINAL DICT FOR MIMIC-{data_type} ***'): # III_label.ICUSTAY_ID.unique()
        dict_by_icuid = {}
        lab = lab_data[lab_data[ICUSTAY_ID] == id]
        prescrip = prescrip_data[prescrip_data[ICUSTAY_ID] == id]
        input_ = input_data[input_data[ICUSTAY_ID] == id]
        
        data = MIMIC_make_charttime_data(data_type, lab, prescrip, input_)
        
        dict_by_icuid['icustay_id'] = id
        dict_by_icuid['label'] = label_data[label_data[ICUSTAY_ID]==id].labels.iloc[0]
        dict_by_icuid['intime'] = icustay_data[icustay_data[ICUSTAY_ID]==id][INTIME].iloc[0]
        dict_by_icuid['data'] = data
        
        final.append(dict_by_icuid)
        
    return final


def EICU_make_data(lab, prescrip, input_):
    '''
    Data using lab, prescrip, input by id (no charttime)
    '''
    # lab_columns = ['labresultoffset', 'labtypeid', 'labname', 'labresult', 'labresulttext', 'labmeasurename', 'labresultrevisedoffset']
    # prescrip_columns = ['drugorderoffset', 'drugstartoffset', 'drugivadmixture', 'drugordercancelled', 'drugname', 'drughiclseqno', 'dosage', 'routeadmin', 'frequency', 'loadingdose', 'prn', 'drugstopoffset', 'gtc']
    # input_columns = ['infusionoffset', 'drugname', 'drugrate', 'infusionrate', 'drugamount', 'volumeoffluid', 'patientweight']
    
    lab_columns = list(lab.columns)
    prescrip_columns = list(prescrip.columns)
    input_columns = list(input_.columns)
    
    data = {}
    labs = []
    prescrips = []
    inputs = []
    
    for i in range(len(lab)):
        by_lab={}
        for col in lab_columns:
            by_lab[col] = lab[col].iloc[i]
        labs.append(by_lab)
        
    for i in range(len(prescrip)):
        by_prescrip={}
        for col in prescrip_columns:
            by_prescrip[col] = prescrip[col].iloc[i]
        prescrips.append(by_prescrip)
        
    for i in range(len(input_)):
        by_input={}
        for col in input_columns:
            by_input[col] = input_[col].iloc[i]
        inputs.append(by_input)

    data['labs'] = labs
    data['prescrips'] = prescrips
    data['inputs'] = inputs

    return data


def EICU_make_final_dict(lab_data, prescrip_data, input_data, label_data):
    '''
    Create final dictionary by collecting data corresponding to each id
    '''
    final = []
    for id in tqdm(label_data['patientunitstayid'].unique(), desc= f'*** MAKE FINAL DICT FOR EICU ***'):
        dict_by_icuid = {}
        
        lab = lab_data[lab_data['patientunitstayid'] == id]
        prescrip = prescrip_data[prescrip_data['patientunitstayid'] == id]
        input_ = input_data[input_data['patientunitstayid'] == id]
    
        data = EICU_make_data(lab, prescrip, input_)
        
        dict_by_icuid['patientunitstayid'] = id
        dict_by_icuid['label'] = label_data[label_data['patientunitstayid']==id].labels.iloc[0]
        dict_by_icuid['data'] = data
        
        final.append(dict_by_icuid)
        
    return final


def main(args):
    """
    TODO:
        Implement your feature preprocessing function here.
        Rename the file name with your student number.
    
    Note:
        1. This script should dump processed features to the --dest directory.
        Note that --dest directory will be an input to your dataset class (i.e., --data_path).
        You can dump any type of files such as json, cPickle, or whatever your dataset can handle.

        2. If you use vocabulary, you should specify your vocabulary file(.pkl) in this code section.
        Also, you must submit your vocabulary file({student_id}_vocab.pkl) along with the scripts.
        Example:
            with open('./20231234_vocab.pkl', 'rb') as f:
                (...)
    """

    root_dir = args.root
    dest_dir = args.dest
    
    #### Load Data
    III_DATA_PATH =  root_dir + 'mimiciii/'
    III_lab = pd.read_csv(III_DATA_PATH + 'LABEVENTS.csv')
    III_prescrip = pd.read_csv(III_DATA_PATH + 'PRESCRIPTIONS.csv')
    III_input_cv = pd.read_csv(III_DATA_PATH + 'INPUTEVENTS_CV.csv')
    III_input_mv = pd.read_csv(III_DATA_PATH + 'INPUTEVENTS_MV.csv')
    III_icustay = pd.read_csv(III_DATA_PATH + 'ICUSTAYS.csv')
    III_input_items = pd.read_csv(III_DATA_PATH + 'D_ITEMS.csv')
    III_lab_items = pd.read_csv(III_DATA_PATH + 'D_LABITEMS.csv')
    III_label = pd.read_csv(root_dir + 'labels/mimiciii_labels.csv')
    
    IV_DATA_PATH = root_dir + 'mimiciv/'
    IV_lab = pd.read_csv(IV_DATA_PATH + 'labevents.csv')
    IV_prescrip = pd.read_csv(IV_DATA_PATH + 'prescriptions.csv')
    IV_input = pd.read_csv(IV_DATA_PATH + 'inputevents.csv')
    IV_icustay = pd.read_csv(IV_DATA_PATH + 'icustays.csv')
    IV_input_items = pd.read_csv(IV_DATA_PATH + 'd_items.csv')
    IV_lab_items = pd.read_csv(IV_DATA_PATH + 'd_labitems.csv')
    IV_label = pd.read_csv(root_dir + 'labels/mimiciv_labels.csv')

    EICU_DATA_PATH = root_dir + 'eicu/'
    EICU_lab = pd.read_csv(EICU_DATA_PATH + 'lab.csv')
    EICU_prescrip = pd.read_csv(EICU_DATA_PATH + 'medication.csv')
    EICU_input = pd.read_csv(EICU_DATA_PATH + 'infusionDrug.csv')
    EICU_label = pd.read_csv(root_dir + 'labels/eicu_labels.csv')

        
    ### Create ICUSTAY_ID column for MIMIC Dataset
    III_lab['ICUSTAY_ID'] = int(0)
    IV_lab['stay_id'] = int(0)
    IV_prescrip['stay_id'] = int(0)
    III_lab, IV_lab, IV_prescrip  = make_icustay_id(III_lab, III_icustay, IV_lab, IV_prescrip, IV_icustay) # 17m + 26m + 11m

    ### Concat the inputCV+inputMV for MIMIC-III
    III_input_mv.rename(columns = {'STARTTIME' : 'CHARTTIME'}, inplace = True)
    III_input = pd.concat([III_input_cv, III_input_mv], axis=0).sort_values('ICUSTAY_ID')
    III_input = III_input.reset_index(drop=True)

    # ### Look-up table for MIMIC dataset
    III_input = look_up_table('III', III_input, III_input_items, 1) # 22m
    III_lab = look_up_table('III', III_lab, III_lab_items, 2) # 14m
    IV_input = look_up_table('IV', IV_input, IV_input_items, 3) # 11m
    IV_lab = look_up_table('IV', IV_lab, IV_lab_items, 4) # 23m
    
    ### Merge two similar columns for eICU
    EICU_lab['labmeasurename'] = np.where(pd.notnull(EICU_lab['labmeasurenamesystem']) == True, 
                                          EICU_lab['labmeasurenamesystem'], EICU_lab['labmeasurenameinterface'])
    EICU_lab.drop(['labmeasurenamesystem', 'labmeasurenameinterface'], axis=1, inplace=True)
    
    ### Make dictionary
    III_final = MIMIC_make_final_dict('III', III_lab, III_prescrip, III_input, III_label, III_icustay) # 18m
    IV_final = MIMIC_make_final_dict('IV', IV_lab, IV_prescrip, IV_input, IV_label, IV_icustay) # 26m
    EICU_final = EICU_make_final_dict(EICU_lab, EICU_prescrip, EICU_input, EICU_label) # 12m
    
    ### Save
    if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
    with open(dest_dir + 'mimiciii_preprocessed.pickle', 'wb') as f:
        pickle.dump(III_final, f)
    with open(dest_dir + 'mimiciv_preprocessed.pickle', 'wb') as f:
        pickle.dump(IV_final, f)
    with open(dest_dir + 'eicu_preprocessed.pickle', 'wb') as f:
        pickle.dump(EICU_final, f)



if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)