"""

"""

from __future__ import annotations
import csv
from dataclasses import dataclass
from enum import IntEnum, auto
import itertools
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def pairwise(it: Iterator[T]) -> Iterator[Tuple[Optional[T], T]]:
    prev = None
    cur = None
    for cur in it:
        yield prev, cur
        prev = cur
    yield prev, cur


def lexer(infile):
    reader = csv.reader(infile)

    mode: Union[None, Literal['epoch'], Literal['batch'], Literal['events']] = None

    for row in reader:
        if row == ['=== EPOCH ===']:
            mode = 'epoch'
        elif row == ['=== BATCH ===']:
            mode = 'batch'
        elif row == ['=== EVENTS ===']:
            mode = 'events'
        else:
            yield mode, row


class EpochColumns(IntEnum):
    dataset = 0
    div = auto()
    events = auto()
    num_conv_layers = auto()
    rank = auto()
    size = auto()
    epoch = auto()
    seconds = auto()
    loss = auto()
    accuracy = auto()


class EventsColumns(IntEnum):
    id = 0
    events = auto()


@dataclass
class Event:
    nepochs: int
    nworkers: int
    batch: int
    reload: bool
    checkpoint: bool


def main(outfile, infile):
    Xs = []
    Ys = []
    Es = []
    for mode, rows in itertools.groupby(lexer(infile), key=lambda x: x[0]):
        if mode == 'epoch':
            X = []
            Y = []
            E = None

            _ = next(rows)  # header

            for _, row in rows:
                x = float(row[EpochColumns.epoch])
                y = float(row[EpochColumns.seconds])
                e = len(Es)

                X.append(x)
                Y.append(y)
                E = e

            if len(X) == 0:
                continue

            Xs.append(X)
            Ys.append(Y)
            Es.append(E)

        elif mode == 'events' and False:
            lookup = {}

            _ = next(rows)  # header
            for _, row in rows:
                id = row[EventsColumns.id]
                events = row[EventsColumns.events]

                events = eval(events, {'__builtins__': None}, {'Event': Event})

                ckpt_freq = 0
                last_ckpt = None
                for event in events:
                    ckpt_freq += event.nepochs
                    if event.checkpoint:
                        if last_ckpt is None:
                            last_ckpt = event
                        elif last_ckpt is not None:
                            break

                failure_epoch = 0
                for event in events:
                    if event.reload:
                        break

                    failure_epoch += event.nepochs

                lookup[id] = f'Checkpoint every {ckpt_freq} epochs; Failure at {failure_epoch} epoch'


        else:
            # consume
            for row in rows:
                pass

    lookup = {}
    for id, (ckpt_freq, failure_epoch) in enumerate(itertools.product((1, 2, 4, 8, 16), range(4, 65, 8))):
        lookup[id] = f'Checkpoint every {ckpt_freq} epochs; Failure at {failure_epoch} epoch'
        
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    for X, Y, E in zip(Xs, Ys, Es):
        X = X
        Y = np.cumsum(Y)
        #E = lookup[E]
        ax.plot(X, Y)
        ax.scatter((X[-1],), (Y[-1],), color='r')
        ax.set_xlabel('Epoch Number')
        #ax.set_ylabel('Training Loss')
        #ax.set_title('Training Loss vs Epoch Number')
    fig.savefig(outfile)


def cli():
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', '-o', required=True, type=argparse.FileType('wb'))
    parser.add_argument('infile', default=sys.stdin, type=argparse.FileType('r'))
    args = vars(parser.parse_args())

    main(**args)


if __name__ == '__main__':
    cli()
