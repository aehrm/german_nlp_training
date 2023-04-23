from tagger.rwtagger.tagger import RWTagger
import pandas
import logging
import sys

for kind in ['direct', 'indirect', 'freeIndirect', 'reported']:
    print('training ', kind)
    train_df = pandas.read_csv(f'./data_konvens-paper-2020/train/{kind}_combined.tsv', sep='\t')
    val_df = pandas.read_csv(f'./data_konvens-paper-2020/val/{kind}_combined.tsv', sep='\t')
    test_df = pandas.read_csv(f'./data_konvens-paper-2020/test/{kind}_combined.tsv', sep='\t')

    os.path.mkdir(f'./train_output/{kind}/')
    tagger = RWTagger(device='cuda:0')
    tagger.train(train_df, val_df, test_df, f'./train_output/{kind}/', embtype=sys.argv[1])
