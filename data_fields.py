DATA_FIELDS = {
    'ADMISSIONS' : ['HADM_ID', 'SUBJECT_ID', 'ADMITTIME', 'DISCHTIME', 'DEATHTIME', 'ADMISSION_TYPE', 'LANGUAGE', 'RELIGION',
                    'MARITAL_STATUS', 'ETHNICITY', 'EDREGTIME', 'EDOUTTIME'],
    'OUTPUTEVENTS' : ['ICUSTAY_ID', 'CHARTTIME', 'VALUE', 'VALUEUOM', 'STORETIME', 'STOPPED', 'NEWBOTTLE', 'ISERROR'],
    'CHARTEVENTS' : ['ICUSTAY_ID', 'CHARTTIME', 'STORETIME', 'VALUE', 'VALUENUM', 'VALUEUOM', 'WARNING', 'ERROR', 'RESULTSTATUS',
                     'STOPPED'],
    'PROCEDURES_ICD' : ['SEQ_NUM', 'ICD9_CODE'],
    'MICROBIOLOGYEVENTS' : ['ICUSTAY_ID', 'CHARTTIME', 'SPEC_TYPE_DESC', 'ORG_NAME', 'AB_NAME', 'ISOLATE_NUM', 'DILUTION_TEXT',
                            'DILUTION_COMPARISON', 'DILUTION_VALUE', 'INTERPRETATION'],
    'LABEVENTS' : ['ROW_ID', 'ICUSTAY_ID', 'CHARTTIME', 'VALUE', 'VALUENUM', 'VALUEUOM', 'FLAG'],
    'DIAGNOSES_ICD' : ['SEQ_NUM', 'ICD9_CODE'],
    'NOTEEVENTS' : ['ICUSTAY_ID', 'CHARTDATE', 'CATEGORY', 'DESCRIPTION', 'ISERROR', 'TEXT', 'CHARTTIME'],
    'PRESCRIPTIONS' : ['ICUSTAY_ID', 'STARTDATE', 'DRUG_TYPE', 'DRUG', 'DRUG_NAME_POE', 'DRUG_NAME_GENERIC', 'GSN',
                       'FORMULARY_DRUG_CD', 'NDC', 'PROD_STRENGTH', 'DOSE_VAL_RX', 'DOSE_UNIT_RX', 'FORM_VAL_DISP',
                       'FORM_UNIT_DISP', 'ROUTE'],
    'CPTEVENTS' : ['CHARTDATE', 'CPT_CD', 'CPT_NUMBER', 'CPT_SUFFIX', 'SECTIONHEADER', 'SUBSECTIONHEADER',
                   'DESCRIPTION'],
    'INPUTEVENTS_CV' : ['ICUSTAY_ID', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM', 'STORETIME', 'CHARTTIME', 'STOPPED', 'NEWBOTTLE'],
    'INPUTEVENTS_MV' : ['ICUSTAY_ID', 'STARTTIME', 'ENDTIME', 'AMOUNT', 'AMOUNTUOM', 'RATE', 'RATEUOM', 'CANCELREASON']
}

D_ITEMS_RELATION = {
    'INPUTEVENTS_CV' : {
        'ITEMID' : {
            'ITEMID' : 'ITEMID',
            'LABEL' : 'ITEM'
        }
    },
    'INPUTEVENTS_MV' : {
        'ITEMID' : {
            'ITEMID' : 'ITEMID',
            'LABEL' : 'ITEM'
        }
    },
    'OUTPUTEVENTS' : {
        'ITEMID' : {
            'ITEMID' : 'ITEMID',
            'LABEL' : 'ITEM'
        },
    },
    'CHARTEVENTS' : {
        'ITEMID' : {
            'ITEMID' : 'ITEMID',
            'LABEL' : 'ITEM'
        },
    },
    'MICROBIOLOGYEVENTS': {
        'SPEC_ITEMID' : {
            'ITEMID' : 'SPEC_ITEMID',
            'LABEL' : 'SPECIMEN'
        },
        'ORG_ITEMID' : {
            'ITEMID' : 'ORG_ITEMID',
            'LABEL' : 'ORGANISM'
        },
        'AB_ITEMID' : {
            'ITEMID' : 'AB_ITEMID',
            'LABEL' : 'ANTIBODY'
        },
    },
}

D_LABITEMS_RELATION = {
    'LABEVENTS' : {
        'ITEMID' : {
            'ITEMID' : 'ITEMID',
            'LABEL' : 'ITEM',
            'FLUID' : 'FLUID',
            'CATEGORY' : 'CATEGORY',
            'LOINC_CODE' : 'LOINC_CODE'
        },
    },
}