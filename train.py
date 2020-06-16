import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

# from ptb import PTB
from utils import to_var, idx2word, expierment_name
from model import SentenceVAE
from dataset import BERTDataset, WordVocab

def main(args):

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    print("Loading Vocab", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset", args.train_dataset)
    train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.max_sequence_length,
                                corpus_lines=args.corpus_lines, on_memory=args.on_memory)

    print("Loading Test Dataset", args.test_dataset)
    test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.max_sequence_length, on_memory=args.on_memory) \
        if args.test_dataset is not None else None

    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    splits = ['train', 'test']
    data_loaders = {
        'train': train_data_loader,
        'test': test_data_loader
    }

    model = SentenceVAE(
        vocab_size=len(vocab),
        sos_idx=vocab.sos_index,
        eos_idx=vocab.eos_index,
        pad_idx=vocab.pad_index,
        unk_idx=vocab.unk_index,
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args,ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path)
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    NLL = torch.nn.NLLLoss(size_average=False, ignore_index=vocab.pad_index)
    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0):

        # cut-off unnecessary padding from target, and flatten
        
        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):

        for split in splits:

            data_loader = data_loaders[split]

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            correct = 0
            close = 0
            total = 0
            for iteration, batch in enumerate(data_loader):

                batch_size = batch['input'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Forward pass
                logp, mean, logv, z = model(batch['input'], batch['raw_length'])

                # loss calculation
                NLL_loss, KL_loss, KL_weight = loss_fn(logp, batch['target'],
                    batch['raw_length'], mean, logv, args.anneal_function, step, args.k, args.x0)

                loss = (NLL_loss + KL_weight * KL_loss)/batch_size

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                correct += logp.argmax(dim=1).eq(batch['target']).sum().item()
                close += torch.mul(logp.argmax(dim=1).ge(batch["target"]-10), logp.argmax(dim=1).le(batch["target"]+10)).sum().item()
                total += batch['target'].nelement()


                # bookkeepeing
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.view(1,)))

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO"%split.upper(), loss.data[0], epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/NLL Loss"%split.upper(), NLL_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss"%split.upper(), KL_loss.data[0]/batch_size, epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight"%split.upper(), KL_weight, epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f"
                        %(split.upper(), iteration, len(data_loader)-1, loss.item(), NLL_loss.item()/batch_size, KL_loss.item()/batch_size, KL_weight))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['raw'].data, i2w=datasets['train'].get_i2w(), pad_idx=datasets['train'].pad_idx)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            print("%s Epoch %02d/%i, Mean ELBO %9.4f, acc %f, clo %f"%(split.upper(), epoch, args.epochs, torch.mean(tracker['ELBO']), correct/total, close/total))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO"%split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents':tracker['target_sents'], 'z':tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json'%epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch"%(epoch))
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s"%checkpoint_path)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=20)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-pv','--print_every', type=int, default=50)
    parser.add_argument('-tb','--tensorboard_logging', action='store_true')
    parser.add_argument('-log','--logdir', type=str, default='logs')
    parser.add_argument('-bin','--save_model_path', type=str, default='bin')

    parser.add_argument("-c", "--train_dataset", required=True, type=str, help="train dataset for train bert")
    parser.add_argument("-t", "--test_dataset", required=True, type=str, help="test dataset for train bert")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="built vocab model path with bert-vocab")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size") 

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    main(args)
