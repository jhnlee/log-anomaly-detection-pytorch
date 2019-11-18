import os
import pandas as pd
from datetime import datetime


class HyperParamWriter:
    def __init__(self, dir):
        self.dir = dir
        self.hparams = None
        self.load()
        self.writer = dict()

    def update(self, args, tr_loss, tr_acc, val_loss, val_acc):
        now = datetime.now()
        date = '%s-%s-%s %s:%s' % (now.year, now.month, now.day, now.hour, now.minute)
        self.writer.update({'date': date})

        self.writer.update(
            {
                'train_loss': tr_loss,
                'train_accuracy': tr_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
            }
        )

        self.writer.update(vars(args))

        if self.hparams is None:
            self.hparams = pd.DataFrame(self.writer, index=[0])
        else:
            self.hparams = self.hparams.append(self.writer, ignore_index=True)
        self.save()

    def save(self):
        assert self.hparams is not None
        self.hparams.to_csv(self.dir, index=False)

    def load(self):
        path = os.path.split(self.dir)[0]
        if not os.path.exists(path):
            os.mkdir(path)
            self.hparams = None
        elif os.path.exists(self.dir):
            self.hparams = pd.read_csv(self.dir)
        else:
            self.hparams = None

