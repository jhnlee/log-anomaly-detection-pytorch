from torch.utils.data import Dataset
import pickle
import torch


class LogLoader(Dataset):
    def __init__(self, data_path, vocab):
        self.data_path = data_path
        self.blk_id, self.data, self.lengths = self.data_load()
        # Special tokens 정의
        self.pad, self.bos, self.eos, self.unk = (vocab.index(v) for v in ['<PAD>', '<BOS>', '<EOS>', '<UNK>'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        encoder_input, decoder_input = self.preprocess(self.data[item])
        return self.blk_id[item], encoder_input, decoder_input, self.lengths[item]

    def data_load(self):
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        blk, logs = list(zip(*data.items()))
        lengths = [len(d) + 1 for d in logs]
        return blk, logs, lengths

    def preprocess(self, log):
        # Special token 추가를 위해 input 조정
        log = [int(l) + 4 for l in log]
        # eos, bos 추가
        encoder_input, decoder_input = log + [self.eos], [self.bos] + log
        return encoder_input, decoder_input

    def padding(self, log, max_len):
        return [l + [self.pad] * (max_len - len(l)) for l in log]

    def batch_sequence(self, batch):
        blk_id, encoder_input, decoder_input, lengths = [d for d in list(zip(*batch))]
        assert len(encoder_input) == len(decoder_input) and len(encoder_input) == len(lengths)

        max_seq_length = max(lengths)

        # add pad tokens and make mask for attention
        encoder_input, decoder_input = torch.tensor([self.padding(log, max_seq_length)
                                                     for log in (encoder_input, decoder_input)])
        input_mask = torch.ones_like(encoder_input).masked_fill(encoder_input == 0, 0)
        assert len(decoder_input[0]) == max_seq_length

        return blk_id, input_mask, encoder_input, decoder_input

