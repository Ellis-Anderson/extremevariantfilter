import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelBinarizer


# Common Functions


def Open_VCF(vcf_path):
    header=['CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT', 'CALLS']
    vcf = pd.read_csv(vcf_path, delimiter="\t", comment="#", names=header)
    return vcf


def Split_Info(info):
    fields = ['QD=', 'MQ=', 'MQRankSum=', 'FS=', 'ReadPosRankSum=', 'SOR=']
    parts = dict(part.split('=') for part in info.split(';') if any(field in part for field in fields))
    return parts


def Get_Calls_Info(vcf):
    call_fields = vcf['CALLS'].str.split(":", expand=True)
    call_fields.columns = vcf['FORMAT'][0].split(":")
    GTS = pd.get_dummies(call_fields['GT'])
    AD = call_fields['AD'].str.split(',', expand=True)
    AD.columns = ['RefD', 'AltD', 'AltAltD']
    AD = AD.drop('AltAltD', axis=1)
    AD['RefD'] = pd.to_numeric(AD['RefD'])
    AD['AltD'] = pd.to_numeric(AD['AltD'])
    AD['RDper'] = (AD['RefD']/(AD['RefD'] + AD['AltD']))
    AD['ADrat'] = (AD['AltD']/(AD['RefD'] + .1))
    calls = pd.concat([GTS['0/1'], AD], axis=1)
    return calls


def check_type(poly):
    if poly != 'SNP' and poly != 'INDEL':
        raise ValueError('--type takes only values SNP or INDEL')

    return


# Apply Filter Functions


def Get_Header(vcf_path):
    with open(vcf_path, 'r') as vcf:
        header = []
        newline = vcf.readline()
        while newline.startswith('#'):
            if 'FILTER' in newline or "CHROM\tPOS" in newline and filter_written == False:
                header.append('##FILTER=<ID=XGB_SNP,Description="Likely FP SNP as determined by loaded model">\n')
                header.append('##FILTER=<ID=XGB_IND,Description="Likely FP InDel as determined by loaded model">\n')
                filter_written = True
            header.append(newline)
            newline = vcf.readline()
    return header


def Check_SNP(vcf):
    if "," in vcf['ALT']:
        if len(vcf['REF']) == 1 and (len(vcf['ALT'].split(',')[0]) == 1 or \
                                     len(vcf['ALT'].split(',')[1]) == 1):
            return 1
        else:
            return 0
    elif len(vcf['REF']) == 1 and len(vcf['ALT']) == 1:
        return 1
    else:
        return 0


def Predict_Var(vcf, snp_mdl, ind_mdl):
    params = vcf.iloc[0:11].values
    if vcf['Is_SNP'] == 1:
        return int(snp_mdl.predict(params[None, :]))
    else:
        return int(ind_mdl.predict(params[None, :]))


def Add_Filter(vcf):
    if vcf['Is_SNP'] == 1 and vcf['Predict'] == 0:
        return "XGB_SNP"
    elif vcf['Is_SNP'] == 0 and vcf['Predict'] == 0:
        return "XGB_IND"
    else:
        return "."


def Get_Name(path):
    filename = path.split('/').pop()
    basename = '.'.join(filename.split('.')[0:-1])
    outname = basename + '.filter.vcf'
    return outname


def Write_VCF(table, header, outname):
    with open(outname, 'w') as vcf_out:
        for line in header:
            vcf_out.write("%s" % line)
    table.to_csv(outname, sep = '\t', header = False, mode = 'a', index = False)


# Train Model Functions


def Make_Table(vcf, label):
    var_vcf = Open_VCF(vcf)
    info_fields = pd.DataFrame(list(var_vcf['INFO'].apply(Split_Info))).fillna(0.)
    info_fields = info_fields[['QD', 'MQ', 'FS', 'MQRankSum', 'ReadPosRankSum', 'SOR']]
    calls = Get_Calls_Info(var_vcf)
    info_fields = pd.concat([info_fields, calls], axis=1)
    info_fields['label'] = label
    return info_fields


def Get_Training_Table(tp_vcf, fp_vcf):
    tp_snp_vcf = Make_Table(tp_vcf, 1)
    fp_snp_vcf = Make_Table(fp_vcf, 0)
    full_vcf = tp_snp_vcf.append(fp_snp_vcf)
    array = full_vcf.values
    X = array[:,0:11]
    Y = array[:,11]
    return X, Y.astype(int)


def Get_Training_Tables(tp_fp_tup):
    tp_vcf, fp_vcf = tp_fp_tup
    tp_snp_vcf = Make_Table(tp_vcf, 1)
    fp_snp_vcf = Make_Table(fp_vcf, 0)
    full_vcf = tp_snp_vcf.append(fp_snp_vcf)
    array = full_vcf.values
    X = array[:,0:11]
    Y = array[:,11]
    return X, Y.astype(int)


def Build_Model(poly, njobs):
    seed = 7
    if poly == "SNP":
        model = ("XGBoost('gbtree', 0.3, 6, 600)",
                 XGBClassifier(n_estimators=600,
                 learning_rate=0.3, max_depth=6, random_state=seed,
                 algorithm='gbtree', objective="binary:logistic",
                 nthread=njobs))
    else:
        model = ("XGBoost('gbtree', 0.3, 6, 1000)",
                 XGBClassifier(n_estimators=1000,
                 learning_rate=0.3, max_depth=6, random_state=seed,
                 algorithm='gbtree', objective="binary:logistic",
                 nthread=njobs))

    return model
