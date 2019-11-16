import os
import argparse
import torch
import random
from utils import HyperParamWriter
from vocab import Vocab
from log_loader import LogLoader
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
from model import GruAutoEncoder
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train(args, device, model, log_vocab):
    # for reproductibility
    set_seed(args)

    # Load Datasets
    tr_set = LogLoader(data_path=args.train_data_path,
                       vocab=log_vocab)

    val_set = LogLoader(data_path=args.val_data_path,
                        vocab=log_vocab)

    tr_loader = DataLoader(dataset=tr_set,
                           batch_size=args.batch_size,
                           shuffle=True,
                           num_workers=args.num_workers,
                           pin_memory=True,
                           drop_last=True,
                           collate_fn=tr_set.batch_sequence)

    val_loader = DataLoader(dataset=val_set,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            drop_last=True,
                            collate_fn=tr_set.batch_sequence)

    # Load optimizer
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.learning_rate)
    total_step = len(tr_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_step)
    warmup_scheduler = GradualWarmupScheduler(optimizer,
                                              multiplier=10,
                                              total_epoch=total_step * args.warmup_percent,
                                              after_scheduler=scheduler)

    # for low-precision training
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # tensorboard
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    writer = SummaryWriter(args.save_path)

    best_val_loss = 1e+9
    global_step = 0

    train_loss = 0
    train_acc = 0

    for epoch in tqdm(range(args.epochs), desc='epocs'):
        for step, batch in tqdm(enumerate(tr_loader), desc='steps', total=len(tr_loader)):
            model.train()
            input_mask, encoder_input, decoder_input = map(lambda x: x.to(device), batch[1:])
            inputs = {
                'encoder_mask': input_mask,
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
            }
            outputs, loss = model(**inputs)
            pred = outputs.max(dim=2)[1].transpose(0, 1)  # (B x L)

            # mean accuracy except for pad token
            not_pad = encoder_input != log_vocab.index('<PAD>')
            num_words = not_pad.sum()
            batch_acc = (pred[not_pad] == encoder_input[not_pad]).float().sum() / num_words

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_clip_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.step()
            model.zero_grad()
            global_step += 1
            warmup_scheduler.step()

            show_lr = warmup_scheduler.get_lr()[0]
            writer.add_scalars('lr', {'lr': show_lr}, global_step)
            if global_step % args.eval_step == 0:
                tqdm.write('global_step: {:3}, '
                           'tr_loss: {:.3f}, '
                           'tr_acc: {:.3f}, '.format(global_step, loss, batch_acc, ))

        # Evaluate at the end of batch
        val_loss, val_acc = evaluate(val_loader, model, log_vocab, device)

        writer.add_scalars('loss', {'train': train_loss / (step + 1),
                                    'val': val_loss}, global_step)
        writer.add_scalars('acc', {'train': train_acc / (step + 1),
                                   'val': val_acc}, global_step)

        tqdm.write('global_step: {:3}, '
                   'tr_loss: {:.3f}, '
                   'val_loss: {:.3f}, '
                   'tr_acc: {:.3f}, '
                   'val_acc: {:.3f} '.format(global_step,
                                             train_loss / (step + 1),
                                             val_loss,
                                             train_acc / (step + 1),
                                             val_acc))

        if val_loss < best_val_loss:
            name = '/bestmodel.bin'
            torch.save(model.state_dict(), args.save_path + name)
            best_val_loss = val_loss
            best_val_acc = val_acc

    writer.close()

    return global_step, train_loss / (step + 1), best_val_loss, train_acc / (step + 1), best_val_acc


def evaluate(dataloader, model, vocab, device):
    val_loss = 0
    val_acc = 0

    for val_step, batch in enumerate(dataloader):
        model.eval()

        input_mask, encoder_input, decoder_input = map(lambda x: x.to(device), batch[1:])

        inputs = {
            'encoder_mask': input_mask,
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
        }

        with torch.no_grad():
            outputs, loss = model(**inputs)

            pred = outputs.max(dim=2)[1].transpose(0, 1)  # (B x 2L)

            # mean accuracy except pad token
            not_pad = encoder_input != vocab.index('<PAD>')
            num_words = not_pad.sum()
            batch_acc = (pred[not_pad] == encoder_input[not_pad]).float().sum() / num_words

            val_loss += loss
            val_acc += batch_acc

    val_loss /= (val_step + 1)
    val_acc /= (val_step + 1)

    return val_loss.item(), val_acc


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--embedding_hidden_dim", default=128, type=int,
                        help="hidden dimension for embedding matrix")
    parser.add_argument("--num_hidden_layer", default=1, type=int,
                        help="number of gru layers in encoder")
    parser.add_argument("--gru_hidden_dim", default=512, type=int,
                        help="hidden dimension for encoder and decoder gru")
    parser.add_argument("--dropout_p", default=0.1, type=float,
                        help="dropout percentage for encoder and decoder gru")
    parser.add_argument("--attention_method", default="dot", type=str,
                        help="attention method (dot, general, concat)")

    # Train parameter
    parser.add_argument("--batch_size", default=512, type=int,
                        help="batch size")
    parser.add_argument("--warmup_percent", default=0.01, type=int,
                        help="Linear warmup over total step * warmup_percent.")
    parser.add_argument("--learning_rate", default=5e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=50, type=int,
                        help="total epochs")
    parser.add_argument("--eval_step", default=1000, type=int,
                        help="show training accuracy on every eval step")
    parser.add_argument("--grad_clip_norm", default=1.0, type=float,
                        help="batch size")

    # Other parameters
    parser.add_argument("--device", default='cuda', type=str,
                        help="Whether to use cpu or cuda")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed(default=0)")

    # Path parameters
    parser.add_argument("--vocab_path", default='./data/vocab.txt', type=str, required=True,
                        help="vocab.txt directory")
    parser.add_argument("--train_data_path", default='./data/train.pkl', type=str, required=True,
                        help="train dataset directory")
    parser.add_argument("--val_data_path", default='./data/val.pkl', type=str, required=True,
                        help="validation dataset directory")
    parser.add_argument("--save_path", default='./data/model/', type=str, required=True,
                        help="directory where model parameters will be saved")

    args = parser.parse_args()

    set_seed(args)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log_vocab = Vocab(args.vocab_path).vocab
    model = GruAutoEncoder(vocab=log_vocab,
                           embedding_hidden_dim=args.embedding_hidden_dim,
                           num_hidden_layer=args.num_hidden_layer,
                           gru_hidden_dim=args.gru_hidden_dim,
                           device=device,
                           dropout_p=args.dropout_p,
                           attention_method=args.attention_method)
    global_step, train_loss, best_val_loss, train_acc, best_val_acc = train(args, device, model, log_vocab)

    # Write hyperparameter
    hyper_param_writer = HyperParamWriter('./hyper_search/hyper_parameter.csv')
    hyper_param_writer.update(args, train_loss, train_acc, best_val_loss, best_val_acc)

