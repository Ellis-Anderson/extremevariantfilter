#!/usr/bin/env python
"""
Usage:
    train_model.py (--true-pos STR) (--false-pos STR) (--type STR) [--out STR] [--njobs INT] [--verbose]

Description:
    Train a model to be saved and used with VCFs.

Arguments:
    --true-pos STR          Path to true-positive VCF from VCFeval or comma-seperated list of paths
    --false-pos STR         Path to false-positive VCF from VCFeval or comma-seperated list of paths
    --type STR              SNP or INDEL

Options:
    -o, --out <STR>                 Outfile name for writing model [default: (type).filter.pickle.dat]
    -n, --njobs <INT>               Number of threads to run in parallel [default: 2]
    -h, --help                      Show this help message and exit.
    -v, --version                   Show version and exit.
    --verbose                       Log output

Examples:
    python train_table.py --true-pos <path/to/tp/vcf> --false-pos <path/to/fp/vcf> --type [SNP, INDEL] --njobs 20
"""

import extremevariantfilter as evf
from multiprocessing import Pool
from contextlib import closing
import warnings


def get_options():
    args = docopt(__doc__, version='1.0')
    verbose = args['--verbose']

    # Read training data
    tp_vcf = args['--true-pos']
    fp_vcf = args['--false-pos']
    poly = args['--type']
    njobs = int(args['--njobs'])
    outname = args['--out']
    if outname == "(type).filter.pickle.dat":
        outname = poly + '.filter.pickle.dat'
    evf.check_type(poly)

    return tp_vcf, fp_vcf, poly, njobs, outname


def main():
    warnings.filterwarnings('ignore',category=DeprecationWarning)

    tp_vcf, fp_vcf, poly, njobs, outname = get_options()
    all_vcf = evf.Check_VCF_Paths(tp_vcf, fp_vcf)
    with closing(Pool(processes=njobs)) as pool:
        results = pool.map(evf.Get_Training_Tables, all_vcf)
        pool.terminate()

    X, Y = zip(*results)
    X_all = np.concatenate(X)
    Y_all = np.concatenate(Y)

    model = evf.Build_Model(poly, njobs)
    print("Training {} on {}").format(model[0], all_vcf)
    model[1].fit(X_all, Y_all)
    with open(outname, "wb") as out:
        pickle.dump(model[1], out)


if __name__ == "__main__":
    main()
