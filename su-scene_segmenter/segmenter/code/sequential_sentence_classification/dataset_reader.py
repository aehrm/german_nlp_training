import itertools
import json
from typing import Dict, List
from overrides import overrides

import numpy as np
import copy

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data import TokenIndexer, Tokenizer
from allennlp.data.instance import Instance
from allennlp.data.fields.field import Field
from allennlp.data.fields import TextField, LabelField, ListField, ArrayField, MultiLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer
from allennlp.data.tokenizers import WhitespaceTokenizer
from allennlp.data.tokenizers.token_class import Token


@DatasetReader.register("SeqClassificationReader")
class SeqClassificationReader(DatasetReader):
    """
    Reads a file from Pubmed-RCT dataset. Each instance contains an abstract_id, 
    a list of sentences and a list of labels (one per sentence).
    Input File Format: Example abstract below:
        {
        "abstract_id": 5337700, 
        "sentences": ["this is motivation", "this is method", "this is conclusion"], 
        "labels": ["BACKGROUND", "RESULTS", "CONCLUSIONS"]
        }
    """

    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 tokenizer: Tokenizer = None,
                 sent_max_len: int = 100,
                 max_sent_per_example: int = 20,
                 use_sep: bool = True,
                 sci_sum: bool = False,
                 use_abstract_scores: bool = True,
                 sci_sum_fake_scores: bool = True,
                 predict: bool = False,
                 ) -> None:
        super().__init__(manual_distributed_sharding=True,
            manual_multiprocess_sharding=True)
        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.sent_max_len = sent_max_len
        self.use_sep = use_sep
        self.predict = predict
        self.sci_sum = sci_sum
        self.max_sent_per_example = max_sent_per_example
        self.use_abstract_scores = use_abstract_scores
        self.sci_sum_fake_scores = sci_sum_fake_scores
        print("*********************************")
        print("start token : ", self._tokenizer.sequence_pair_start_tokens)
        print("middle token : ", self._tokenizer.sequence_pair_mid_tokens)
        print("end token : ", self._tokenizer.sequence_pair_end_tokens)
        print("*********************************")

    def _read(self, file_path: str):
        file_path = cached_path(file_path)
        with open(file_path) as f:
            for line in self.shard_iterable(f):
                json_dict = json.loads(line)
                instances = self.read_one_example(json_dict)
                for instance in instances:
                    yield instance

    def read_one_example(self, json_dict):
        instances = []
        sentences = json_dict["sentences"]

        if not self.predict:
            labels = json_dict["labels"]
        else:
            labels = None

        confidences = json_dict.get("confs", None)

        additional_features = None


        for sentences_loop, labels_loop, confidences_loop, additional_features_loop in  \
                self.enforce_max_sent_per_example(sentences, labels, confidences, additional_features):

            instance = self.text_to_instance(
                sentences=sentences_loop,
                labels=labels_loop,
                confidences=confidences_loop,
                additional_features=additional_features_loop,
                )
            instances.append(instance)
        return instances

    def enforce_max_sent_per_example(self, sentences, labels=None, confidences=None, additional_features=None):
        """
        Splits examples with len(sentences) > self.max_sent_per_example into multiple smaller examples
        with len(sentences) <= self.max_sent_per_example.
        Recursively split the list of sentences into two halves until each half
        has len(sentences) < <= self.max_sent_per_example. The goal is to produce splits that are of almost
        equal size to avoid the scenario where all splits are of size
        self.max_sent_per_example then the last split is 1 or 2 sentences
        This will result into losing context around the edges of each examples.
        """
        if labels is not None:
            assert len(sentences) == len(labels)
        if confidences is not None:
            assert len(sentences) == len(confidences)
        if additional_features is not None:
            assert len(sentences) == len(additional_features)

        if len(sentences) > self.max_sent_per_example and self.max_sent_per_example > 0:

            i = len(sentences) // 2
            l1 = self.enforce_max_sent_per_example(
                    sentences[:i], None if labels is None else labels[:i],
                    None if confidences is None else confidences[:i],
                    None if additional_features is None else additional_features[:i])
            l2 = self.enforce_max_sent_per_example(
                    sentences[i:], None if labels is None else labels[i:],
                    None if confidences is None else confidences[i:],
                    None if additional_features is None else additional_features[i:])
            return l1 + l2
        else:
            return [(sentences, labels, confidences, additional_features)]

    def is_bad_sentence(self, sentence: str):
        if len(sentence) > 10 and len(sentence) < 600:
            return False
        else:
            return True



    def text_to_instance(self,
                         sentences: List[str],
                         labels: List[str] = None,
                         confidences: List[float] = None,
                         additional_features: List[float] = None,
                         predict: bool = False,
                         ) -> Instance:
        if not self.predict and not predict:
            assert len(sentences) == len(labels)
        if confidences is not None:
            assert len(sentences) == len(confidences)
        if additional_features is not None:
            assert len(sentences) == len(additional_features)

        if self.use_sep:
            origin_sent = copy.deepcopy(sentences)
            sentences = self.shorten_sentences(sentences, self.sent_max_len)
    
            max_len=self.sent_max_len
            while len(sentences[0]) > 512:
                n = int((len(sentences[0])-512)/ len(origin_sent))+1
                
                max_len -= n
                sentences = self.shorten_sentences(origin_sent, max_len )
              
            assert len(sentences[0]) <= 512
    
        else:
            tok_sentences = []
            for sentence_text in sentences:
                if len(self._tokenizer.tokenize(sentence_text)) > self.sent_max_len:
                    tok_sentences.append(self._tokenizer.tokenize(sentence_text)[:self.sent_max_len]+self._tokenizer.sequence_pair_end_tokens)
                else:
                    tok_sentences.append(self._tokenizer.tokenize(sentence_text))
            
            sentences = tok_sentences
           
        fields: Dict[str, Field] = {}
        fields["sentences"] = ListField([
                TextField(sentence)
                for sentence in sentences
        ])

        if labels is not None:
            if isinstance(labels[0], list):
                fields["labels"] = ListField([
                        MultiLabelField(label) for label in labels
                    ])
            else:
                # make the labels strings for easier identification of the neutral label
                # probably not strictly necessary
                fields["labels"] = ListField([
                        LabelField(str(label)+"_label") for label in labels
                    ])
        if confidences is not None:
            fields['confidences'] = ArrayField(np.array(confidences))
        if additional_features is not None:
            fields["additional_features"] = ArrayField(np.array(additional_features))

        return Instance(fields)

    def apply_token_indexers(self, instance: Instance) -> None:
        for text_field in instance["sentences"].field_list:
            text_field.token_indexers = self._token_indexers

    def shorten_sentences(self, origin_sent, max_len):
        tokenized_sentences = [self._tokenizer.sequence_pair_start_tokens]
        for s in origin_sent:
            if len(self._tokenizer.tokenize(s)) > (max_len):
                tokenized_sentences.append(self._tokenizer.tokenize(s)[1:(max_len)]+self._tokenizer.sequence_pair_mid_tokens)
            else:
                tokenized_sentences.append(self._tokenizer.tokenize(s)[1:-1]+self._tokenizer.sequence_pair_mid_tokens)
        mid_tok_len = len(self._tokenizer.sequence_pair_mid_tokens)
        return [list(itertools.chain.from_iterable(tokenized_sentences))[:-mid_tok_len]+self._tokenizer.sequence_pair_end_tokens]