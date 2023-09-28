from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer, AutoTokenizer
import pandas
import itertools
import more_itertools
import flair
import torch
import logging
import os
from datetime import datetime

from flair.data import Sentence, Token, Corpus
from flair.trainers import ModelTrainer
from flair.models import SequenceTagger
from flair.embeddings import TransformerWordEmbeddings

flair.device = torch.device('cuda')
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def token_iterator(df_doc):
    for _, row in df_doc.iterrows():
        if row['tok'] == 'EOF':
            continue
        yield row

def gen_inputs(df_doc, max_size=300):
    sentences = more_itertools.split_before(token_iterator(df_doc), lambda x: x['sentstart'] == 'yes')
    for list_of_sentences in more_itertools.constrained_batches(sentences, max_size=max_size):
        yield itertools.chain(*list_of_sentences)

def tsv_to_inputs(path):
    df = pandas.read_csv(path, sep='\t')
    for _, df_doc in df.groupby('file'):

        for input in gen_inputs(df_doc):
            flair_tokens = []
            for i in input:
                flair_tok = Token(str(i['tok']))
                flair_tok.add_label('cat', i['cat'])
                flair_tokens.append(flair_tok)

            yield Sentence(flair_tokens)


train_data = list(tsv_to_inputs('./data_konvens-paper-2020/train/direct_combined.tsv'))
dev_data = list(tsv_to_inputs('./data_konvens-paper-2020/val/direct_combined.tsv'))
test_data = list(tsv_to_inputs('./data_konvens-paper-2020/test/direct_combined.tsv'))
corpus = Corpus(train_data, dev_data, test_data)
tag_dictionary = corpus.make_label_dictionary(label_type="cat")
logger.info('done reading corpus')

timestamp = datetime.now().isoformat(timespec='seconds')
for kind in ['direct', 'indirect', 'freeIndirect', 'reported']: 
    embeddings = TransformerWordEmbeddings(
            os.getenv('MODEL'), # which transformer model
            layers="-1", # which layers (here: only last layer when fine-tuning)
            pooling_operation='first_last', # how to pool over split tokens
            fine_tune=True, # whether or not to fine-tune
    )

    outdir = os.getenv('OUTPUT', '/output/') + f'/{timestamp}/{kind}/'
    os.makedirs(outdir, exist_ok=True)
    tagger = SequenceTagger(hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=tag_dictionary,
            tag_type="cat",
            use_crf=False,
            use_rnn=False,
    )

    # use Adam optimizer when fine-tuning
    trainer = ModelTrainer(tagger, corpus)

    # fine-tune with setting from BERT paper
    trainer.train(outdir,
        learning_rate=3e-5, # very low learning rate
        optimizer=torch.optim.Adam,
        mini_batch_chunk_size=2, # set this if you get OOM errors
        max_epochs=100, # very few epochs of fine-tuning
    )
