from torch.utils.data import Dataset
import pickle
import torch


class LogLoader(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.blk_id, self.data, self.lengths = self.data_load()
        # Special tokens 정의
        self.pad, self.bos, self.eos, self.unk = 0, 1, 2, 3

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        input_log, output_log = self.preprocess(self.data[item])
        return input_log, output_log, self.lengths[item]

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
        logs = (log + [self.eos], [self.bos] + log)
        return logs

    def padding(self, log, max_len):
        return [l + [self.pad] * (max_len - len(l)) for l in log]

    def batch_sequence(self, batch):
        input_log, output_log, lengths = list(zip(*batch))
        assert len(input_log) == len(output_log)
        assert len(input_log) == len(lengths)

        max_seq_length = max(lengths)

        # pad tokens
        input_log, output_log = [self.padding(log, max_seq_length) for log in (input_log, output_log)]
        assert len(output_log[0]) == max_seq_length

        return (torch.tensor(d) for d in (input_log, output_log, lengths))
