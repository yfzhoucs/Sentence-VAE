from torch.utils.data import Dataset
import tqdm
import torch
import random


class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [[line[:-1].strip(), '']
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2, is_next_label = self.random_sent(item)
        t1_raw, t1_random, t1_label = self.random_word(t1)

        # [CLS] tag = SOS tag, [SEP] tag = EOS tag
        t1 = [self.vocab.sos_index] + t1_random + [self.vocab.eos_index]
        t1_raw = [self.vocab.sos_index] + t1_raw + [self.vocab.eos_index]

        # segment_label = [1 for _ in range(len(t1))]
        bert_input = (t1)[:self.seq_len]
        bert_label = (t1_label)[:self.seq_len]

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_input))]
        input_length = len(bert_input)
        bert_input.extend(padding)
        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(t1_raw))]
        raw_length = len(t1_raw)
        t1_raw.extend(padding)

        padding = [self.vocab.pad_index for _ in range(self.seq_len - len(bert_label))]
        label_length = len(bert_label)
        bert_label = bert_label[0]


        output = {"raw": torch.tensor(t1_raw),
                  "input": torch.tensor(bert_input),
                  "target": torch.tensor(bert_label),
                  "input_length": input_length,
                  "raw_length": raw_length,
                  "label_length": label_length}

        return output

    def random_word(self, sentence):
        raw_tokens = sentence.split()
        tokens = sentence.split()
        input_tokens = []
        target_tokens = []

        for i, token in enumerate(tokens):
            idx = int(token)
            raw_tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
            token = self.vocab.stoi.get(token, self.vocab.unk_index)
            prob = random.random()
            if prob < 0.5 and idx >= 100 and idx < 200 and len(target_tokens) == 0:
                target_tokens.append(token)
            elif len(target_tokens) == 0 and idx >= 100 and idx < 200 and int(tokens[i+1]) >= 200:
                target_tokens.append(token)
            else:
                input_tokens.append(token)

        return raw_tokens, input_tokens, target_tokens

    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)

        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(), 0

    def get_corpus_line(self, item):
        if self.on_memory:
            # print(self.lines[item])
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]


if __name__ == '__main__':
    import argparse
    from  vocab import WordVocab
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser()        
    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("-s", "--seq_len", type=int, default=20, help="maximum sequence len")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="number of batch_size")    
    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

    for x in train_data_loader:
        print(x)
        input()