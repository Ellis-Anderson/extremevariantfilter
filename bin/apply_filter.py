#!/usr/bin/env python
"""
Usage:
    apply_filter.py (--vcf STR) (--snp-model STR) (--indel-model STR) [--verbose]

Description:
    Apply models from train_model.py to a vcf

Arguments:
    --vcf STR                     VCF to be filtered
    --snp-model STR               Model for applying to SNPs
    --indel-model INT             Model for applying to InDels

Options:
    -h, --help                      Show this help message and exit.
    -v, --version                   Show version and exit.
    --verbose                       Log output

Examples:
    python apply_filter.py --vcf <table> --snp-model <snp.pickle.dat> --indel-model <indel.pickle.dat>
"""
import extremevariantfilter as evf
import warnings


def get_options():
    args = docopt(__doc__, version='1.0')
    verbose = args['--verbose']

    # Read training data
    vcf = args['--vcf']
    snp_mod = args['--snp-model']
    ind_mod = args['--indel-model']

    return vcf, snp_mod, ind_mod


def main():
    warnings.filterwarnings('ignore',category=DeprecationWarning)

    vcf_path, snp_mod, ind_mod = get_options()
    header = evf.Get_Header(vcf_path)
    vcf = evf.Open_VCF(vcf_path)

    with open(snp_mod, "rb") as snp_m:
        snp_mdl = pickle.load(snp_m)
    with open(ind_mod, "rb") as ind_m:
        ind_mdl = pickle.load(ind_m)

    info_fields = pd.DataFrame(list(vcf['INFO'].apply(evf.Split_Info))).fillna(0.)
    info_fields = info_fields[['QD', 'MQ', 'FS', 'MQRankSum',
                               'ReadPosRankSum', 'SOR']]
    calls = evf.Get_Calls_Info(vcf)
    info_fields = pd.concat([info_fields, calls], axis=1)
    info_fields['Is_SNP'] = vcf.apply(evf.Check_SNP, axis=1)
    info_fields['Predict'] = info_fields.apply(evf.Predict_Var,
                                               axis=1,
                                               args=(snp_mdl, ind_mdl))
    vcf['FILTER'] = info_fields.apply(evf.Add_Filter, axis=1)

    evf.Write_VCF(vcf, header, Get_Name(vcf_path))


if __name__ == "__main__":
    main()
