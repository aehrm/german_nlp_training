from tagger.rwtagger.tagger import RWTagger
import pandas
import logging
import sys
import os
from datetime import datetime

timestamp = datetime.now().isoformat(timespec='seconds')

for kind in ['direct', 'indirect', 'freeIndirect', 'reported']:
    print('training ', kind)
    train_df = pandas.read_csv(f'./data_konvens-paper-2020/train/{kind}_combined.tsv', sep='\t')
    val_df = pandas.read_csv(f'./data_konvens-paper-2020/val/{kind}_combined.tsv', sep='\t')
    test_df = pandas.read_csv(f'./data_konvens-paper-2020/test/{kind}_combined.tsv', sep='\t')

    os.makedirs(f'/output/{timestamp}/{kind}/', exist_ok=True)
    tagger = RWTagger(device='cuda:0')
    tagger.train(train_df, val_df, test_df, f'/output/{timestamp}/{kind}', batch_len=32, mini_batch_chunk_size=4, chunk_len=int(os.getenv('SEGMENT_LENGTH', '64')) ,embtype='bert:'+os.getenv('MODEL'))
