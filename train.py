import os
import argparse
import random
import logging
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from warmup_scheduler import GradualWarmupScheduler
from datasets import LogLoader
from model.model_ae import GruAutoEncoder

from utils.hyperparam_writer import HyperParamWriter
from utils.vocab import Vocab
from utils.logger import get_logger

logger = get_logger('GRU AutoEncoder')
logger.setLevel(logging.INFO)


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
                                              multiplier=100,
                                              total_epoch=total_step * args.warmup_percent,
                                              after_scheduler=scheduler)

    # for low-precision training
    if args.fp16:
        try:
            from apex import amp
            logger.info('Use fp16')
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # tensorboard 경로 설정
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    writer = SummaryWriter(args.save_path)

    best_val_loss = 1e+9
    global_step = 0

    for epoch in tqdm(range(args.epochs), desc='epochs'):
        train_loss = 0
        train_acc = 0
        for step, batch in tqdm(enumerate(tr_loader), desc='steps', total=len(tr_loader)):
            model.train()
            input_mask, encoder_input, decoder_input = map(lambda x: x.to(device), batch[1:])

            # kwargs를 통해 모델에 input을 넣는 방법
            inputs = {
                'encoder_mask': input_mask,
                'encoder_input': encoder_input,
                'decoder_input': decoder_input,
            }
            outputs, loss = model(**inputs)
            pred = outputs.max(dim=2)[1].transpose(0, 1)  # (B x L)

            # mean accuracy except for pad token (패딩을 제외하고 accuracy 계산)
            not_pad = encoder_input != log_vocab.index('<PAD>')
            num_words = not_pad.sum()
            batch_acc = (pred[not_pad] == encoder_input[not_pad]).float().sum() / num_words

            # mixed precision을 이용하는 경우
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.grad_clip_norm)
            else:
                # 일반적인 경우
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)

            optimizer.step()
            model.zero_grad()
            global_step += 1
            warmup_scheduler.step()

            show_lr = warmup_scheduler.get_lr()[0]
            writer.add_scalars('lr', {'lr': show_lr}, global_step)

            train_loss += loss.item()
            train_acc += batch_acc.item()

            # 학습 상황 확인용
            if global_step % args.eval_step == 0:
                tqdm.write('global_step: {:3}, '
                           'tr_loss: {:.3f}, '
                           'tr_acc: {:.3f}, '
                           'lr: {:.3f}'.format(global_step, loss.item(), batch_acc.item(), show_lr))

                writer.add_scalars('loss', {'train': train_loss / (step + 1)}, global_step)
                writer.add_scalars('acc', {'train': train_acc / (step + 1)}, global_step)

                train_loss = 0
                train_acc = 0

        # Evaluate at the end of step (아래 정의된 evaluate 함수를 통해 validation 수행)
        val_loss, val_acc = evaluate(val_loader, model, log_vocab, device)

        writer.add_scalars('loss', {'val': val_loss}, global_step)
        writer.add_scalars('acc', {'val': val_acc}, global_step)

        tqdm.write('global_step: {:3}, '
                   'val_loss: {:.3f}, '
                   'val_acc: {:.3f} '.format(global_step,
                                             val_loss,
                                             val_acc))

        if val_loss < best_val_loss:
            name = '/bestmodel.bin'
            torch.save(model.state_dict(), args.save_path + name)
            best_val_loss = val_loss
            best_val_acc = val_acc

    writer.close()

    return global_step, train_loss, best_val_loss, train_acc, best_val_acc


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

            pred = outputs.max(dim=2)[1].transpose(0, 1)  # (B x L)

            # mean accuracy except pad token
            not_pad = encoder_input != vocab.index('<PAD>')
            num_words = not_pad.sum()
            batch_acc = (pred[not_pad] == encoder_input[not_pad]).float().sum() / num_words

            val_loss += loss.item()
            val_acc += batch_acc.item()

    val_loss /= (val_step + 1)
    val_acc /= (val_step + 1)

    return val_loss, val_acc


def set_seed(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    # Model parameters
    parser.add_argument("--embedding_hidden_dim", default=64, type=int,
                        help="hidden dimension for embedding matrix")
    parser.add_argument("--num_hidden_layer", default=1, type=int,
                        help="number of gru layers in encoder")
    parser.add_argument("--gru_hidden_dim", default=256, type=int,
                        help="hidden dimension for encoder and decoder gru")
    parser.add_argument("--dropout_p", default=0.1, type=float,
                        help="dropout percentage for encoder and decoder gru")
    parser.add_argument("--attention_method", default="dot", type=str,
                        help="attention method (dot, general, concat)")

    # Train parameter
    parser.add_argument("--batch_size", default=512, type=int,
                        help="batch size")
    parser.add_argument("--warmup_percent", default=0.001, type=float,
                        help="Linear warmup over total step * warmup_percent.")
    parser.add_argument("--learning_rate", default=1e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--epochs", default=3, type=int,
                        help="total epochs")
    parser.add_argument("--eval_step", default=50, type=int,
                        help="show training accuracy on every eval step")
    parser.add_argument("--grad_clip_norm", default=1.0, type=float,
                        help="batch size")

    # Other parameters
    parser.add_argument("--num_workers", default=8, type=int,
                        help="num of working cpu threads")
    parser.add_argument("--device", default='cuda', type=str,
                        help="Whether to use cpu or cuda")
    parser.add_argument('--fp16', default=0, type=int,
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--seed", default=0, type=int,
                        help="Random seed(default=0)")

    # Path parameters
    parser.add_argument("--vocab_path", type=str,  default='./data/vocab.txt',
                        help="vocab.txt directory")
    parser.add_argument("--train_data_path", type=str, default='./data/train.pkl',
                        help="train dataset directory")
    parser.add_argument("--val_data_path", type=str, default='./data/val.pkl',
                        help="validation dataset directory")
    parser.add_argument("--save_path", type=str, default='./model_saved/',
                        help="directory where model parameters will be saved")

    args = parser.parse_args()

    set_seed(args)
    logger.info('training starts.')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log_vocab = Vocab(args.vocab_path).vocab
    model = GruAutoEncoder(vocab=log_vocab,
                           embedding_hidden_dim=args.embedding_hidden_dim,
                           num_hidden_layer=args.num_hidden_layer,
                           gru_hidden_dim=args.gru_hidden_dim,
                           device=device,
                           dropout_p=args.dropout_p,
                           attention_method=args.attention_method).to(device)
    import time
    t = time.time()
    global_step, train_loss, best_val_loss, train_acc, best_val_acc = train(args, device, model, log_vocab)
    elapsed = time.time() - t

    logger.info('training done.')
    logger.info('elapsed time: %.3f Hours' % (elapsed / 3600.))
    logger.info('best acc in testset: %.4f' % best_val_acc)
    logger.info('best loss in testset: %.4f' % best_val_loss)
    
    # Write hyperparameter
    hyper_param_writer = HyperParamWriter('./hyper_search/hyper_parameter.csv')
    hyper_param_writer.update(args, train_loss, train_acc, best_val_loss, best_val_acc)


if __name__ == '__main__':
    main()

