"""

"""

from __future__ import annotations
from contextlib import contextmanager
import csv
from dataclasses import dataclass, field
from datetime import datetime
from io import TextIOWrapper
from pathlib import Path
import re
from statistics import mean, stdev, quantiles
import subprocess
import sys
from typing import *

from typing_extensions import get_type_hints, get_origin, get_args, Annotated


@contextmanager
def subprocess_as_stdin(args):
    process = subprocess.Popen(
        args=args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    old_stdin = sys.stdin
    sys.stdin = process.stdout

    try:
        yield
    finally:
        process.stdout.close()
        process.terminate()
        process.wait(timeout=1)
        process.kill()
        process.wait()

        sys.stdin = old_stdin



@dataclass
class TreeValueProxy:
    tree: Tree

    def __getitem__(self, key) -> str:
        return self.tree[key][1]

    @property
    def logs(self) -> Dict[str, str]:
        return { k: v[1] for k, v in self.tree.logs.items() }

    @property
    def attributes(self) -> Dict[str, datetime]:
        return { k: v[1] for k, v in self.tree.attributes.items() }


@dataclass
class TreeTimestampProxy:
    tree: Tree

    def __getitem__(self, key) -> datetime:
        return self.tree[key][0]

    @property
    def logs(self) -> Dict[str, datetime]:
        return { k: v[0] for k, v in self.tree.logs.items() }

    @property
    def attributes(self) -> Dict[str, datetime]:
        return { k: v[0] for k, v in self.tree.attributes.items() }


@dataclass
class Tree:
    attributes: Dict[str, Tuple[datetime, str]]
    logs: Dict[str, Tuple[datetime, str]]
    children: List[Tree] = field(repr=False)
    parent: Tree
    depth: int = field(init=False)

    def __post_init__(self):
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth + 1

    @property
    def values(self) -> TreeValueProxy:
        return TreeValueProxy(self)

    @property
    def times(self) -> TreeTimestampProxy:
        return TreeTimestampProxy(self)

    def __getitem__(self, key) -> Tuple[datetime, str]:
        if (value := self.attributes.get(f'@{key}', None)) is not None:
            return value
        if (value := self.logs.get(key, None)) is not None:
            return value
        if key == 'children':
            return self.children
        return None

    RE = re.compile(r'''
        ^
        (?:.*)
        (?P<task_id>[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-z0-9]{12})  # e.g. "6a3ad23f-b0c1-47dd-9913-cf537e950839"
        /(?P<task_level>[0-9]+(?:/[0-9]+)*)                                        # e.g. "/8/5/17/5/4"
        [\t]
        (?P<timestamp>[0-9]+\.[0-9]*)                                              # e.g. "1613403872.549264"
        [\t]
        (?P<variable>[^\t]+)                                                       # e.g. "@finished"
        [\t]
        (?P<value>.*)
        $
    ''', re.VERBOSE)

    @classmethod
    def readiter(cls, infiles: List[Path], *, presorted: bool=False) -> Iterator[Tree]:
        if not presorted:
            args = [
                'sort',
                '-t\t',      # tab delimited
                '-k1,1.36',  # sort by first column, first 36 characters (task id)
                '-k1.36V',   # sort by first column, characters after first 36, as a version (task levels)
                '-k2,2n',    # sort by second column, as a number (timestamp)
                *infiles,    # sort these files
            ]
        else:
            args = [
                'sort',
                '-t\t',      # tab delimited
                '-k1,1.36',  # sort by first column, first 36 characters (task id)
                '-k1.36V',   # sort by first column, characters after first 36, as a version (task levels)
                '-k2,2n',    # sort by second column, as a number (timestamp)
                '-m',        # merge already sorted files
                *infiles,    # sort these files
            ]
            
        with subprocess_as_stdin(args):
            last_task_id = None
            last_task_level = None
            last_timestamp = None
            last_variable = None
            last_value = None

            root = None
            tree = None

            for line in sys.stdin:
                line = line.rstrip()
                match = cls.RE.match(line)
                if not match:
                    print( ValueError(f'{line=} does not match {cls.RE=}') )
                    continue

                task_id = match.group('task_id')
                task_level = tuple(map(int, match.group('task_level').split('/')))
                timestamp = datetime.fromtimestamp(float(match.group('timestamp')))
                variable = match.group('variable')
                value = match.group('value')

                #print(f'{task_id=} {task_level=} {timestamp=} {variable=} {value=}')

                if last_task_id is not None and task_id != last_task_id:
                    yield root
                    root = None
                    tree = None

                if tree is None:
                    tree = cls({}, {}, [], None)
                    last_task_id = None
                    last_task_level = None
                    last_timestamp = None
                    last_variable = None
                    last_value = None
                    print('new tree')

                if root is None:
                    root = tree

                if last_task_id is not None and len(task_level) > len(last_task_level):
                    tree = cls({}, {}, [], tree)
                    tree.parent.children.append(tree)

                if last_task_id is not None and len(task_level) < len(last_task_level):
                    tree = tree.parent

                if last_task_id is not None and len(task_level) == len(last_task_level):
                    if task_level[-1] < last_task_level[-1]:
                        tree = cls({}, {}, [], tree.parent)
                        tree.parent.children.append(tree)

                if variable.startswith('@'):
                    tree.attributes[variable] = (timestamp, value)
                else:
                    tree.logs[variable] = (timestamp, value)

                last_task_id = task_id
                last_task_level = task_level
                last_timestamp = timestamp
                last_variable = variable
                last_value = value

        if root is not None:
            yield root
    
    def pprint(self):
        def p(tree, depth=0):
            for attr, (timestamp, value) in tree.attributes.items():
                print(f'{" "*depth}{timestamp} {attr}={value!r}')
            for log, (timestamp, value) in tree.logs.items():
                print(f'{" "*depth}{timestamp} {log}={value!r}')
            for tree in tree.children:
                p(tree, depth+2)

        p(self)
    
    def match(self, pattern):
        class _Temporary:
            p: pattern

        parsed_pattern = get_type_hints(_Temporary, localns={'pattern': pattern}, include_extras=True)['p']
        if get_origin(parsed_pattern) is Annotated:
            parsed_pattern, recursive = get_args(parsed_pattern)
        else:
            recursive = False

        annotations = get_type_hints(parsed_pattern, include_extras=True)
        for name, expected in annotations.items():
            #print(f'{expected=} {self[name]=}')
            expected_any = expected is Any
            got_something = self[name] is not None
            if expected_any and not got_something:
                break

            expected_tuple = get_origin(expected) is tuple
            got_something = self[name] is not None
            if expected_tuple and not got_something:
                break
            elif expected_tuple and got_something:
                exp_timestamp, exp_value = get_args(expected)
                got_timestamp, got_value = self[name]

                #print(f'{exp_timestamp=} {exp_value=}')
                #print(f'{got_timestamp=} {got_value=}')

                expected_any = exp_timestamp is Any
                got_something = got_timestamp is not None
                if expected_any and not got_something:
                    break

                expected_any = exp_value is Any
                got_something = got_value is not None
                if expected_any and not got_something:
                    break

                if get_origin(exp_value) is Literal:
                    exp_value = get_args(exp_value)[0]
                    if exp_value != got_value:
                        break

            expected_list = get_origin(expected) is list
            got_something = self[name] is not None
            if expected_list and not got_something:
                break
            elif expected_list and got_something:
                exp_class = get_args(expected)[0]

                for tree in self.children:
                    if next(tree.match(exp_class), None) is not None:
                        break

        else:
            yield self
            return

        if recursive:
            for tree in self.children:
                yield from tree.match(pattern)



T = TypeVar('T')
Recursive = Annotated[T, True]
NonRecursive = Annotated[T, False]


class BatchPattern:
    started: Tuple[Any, Literal['batch']]
    finished: Tuple[Any, Literal['batch']]
    #epoch: Any
    #loss: Any
    #accuracy: Any


class EpochPattern:
    started: Tuple[Any, Literal['epoch']]
    finished: Tuple[Any, Literal['epoch']]
    #epoch: Any
    #loss: Any
    #accuracy: Any
    #children: List[BatchPattern]


class EventPattern:
    started: Tuple[Any, Literal['event']]
    finished: Tuple[Any, Literal['event']]


class TestPattern:
    started: Tuple[Any, Literal['test']]
    finished: Tuple[Any, Literal['test']]


class ValidPattern:
    started: Tuple[Any, Literal['valid']]
    finished: Tuple[Any, Literal['valid']]


class CheckpointPattern:
    started: Tuple[Any, Literal['checkpoint']]
    finished: Tuple[Any, Literal['checkpoint']]


class ReloadPattern:
    started: Tuple[Any, Literal['reload']]
    finished: Tuple[Any, Literal['reload']]


class TrainPattern:
    started: Tuple[Any, Literal['train']]
    finished: Tuple[Any, Literal['train']]
    #children: List[Recursive[BatchPattern]]


class TrialPattern:
    started: Tuple[Any, Literal['trial']]
    finished: Tuple[Any, Literal['trial']]
    #children: List[Recursive[TrainPattern]]


class TripleRPattern:
    started: Tuple[Any, Literal['triple-r.py']]
    #finished: Tuple[Any, Literal['triple-r.py']]


class MasterPattern:
    started: Tuple[Any, Literal['master']]
    finished: Tuple[Any, Literal['master']]
    #children: List[WorkerPattern]


class WorkerPattern:
    started: Tuple[Any, Literal['worker']]
    finished: Tuple[Any, Literal['worker']]
    #children: List[TripleRPattern]


def main_output_csv(outfile, infiles, presorted):
    writer = csv.writer(outfile)

    event_lookup = {}

    for tree in Tree.readiter(infiles, presorted=presorted):
        Epoch = NewType('Epoch', int)
        Batch = NewType('Batch', int)
        Seconds = NewType('Seconds', float)
        Loss = NewType('Loss', float)
        Accuracy = NewType('Accuracy', float)

        per_epoch: Dict[Tuple[Epoch], Tuple[Seconds, Loss, Accuracy]] = {}
        per_batch: Dict[Tuple[Epoch, Batch], Tuple[Seconds, Loss]] = {}
        parameters = {}

        for trial in tree.match(Recursive[TripleRPattern]):
            print('trial')
            #total_trial_time = (trial.times['finished'] - trial.times['started']).total_seconds()
            #print(f'{total_trial_time = }s')
            parameters = { **parameters, **trial.values.logs }
            for train in trial.match(Recursive[TrainPattern]):
                print('train')
                for epoch in train.match(Recursive[EpochPattern]):
                    print('epoch')
                    per_epoch[(int(epoch.values['epoch']),)] = (
                        (epoch.times['finished'] - epoch.times['started']).total_seconds(),
                        float(epoch.values['loss']),
                        float(epoch.values['accuracy']),
                    )

                    for batch in epoch.children:
                        print('batch')
                        per_batch[int(epoch.values['epoch']), int(batch.values['batch'])] = (
                            (batch.times['finished'] - batch.times['started']).total_seconds(),
                            float(batch.values['loss']),
                        )

        if 'events' in parameters:
            events = parameters['events']
            if events not in event_lookup:
                event_lookup[events] = len(event_lookup)
            parameters['events'] = event_lookup[events]
        
        if 'data_dir' in parameters:
            parameters.pop('data_dir')
        
        if 'checkpoint_dir' in parameters:
            parameters.pop('checkpoint_dir')
        
        if 'hvd.mpi_threads_supported' in parameters:
            parameters.pop('hvd.mpi_threads_supported')
        
        if '_executing_eagerly' in parameters:
            parameters.pop('_executing_eagerly')

        print(parameters)
        parameter_names = sorted(parameters.keys())
        parameter_values = [parameters[k] for k in parameter_names]

        writer.writerow(['=== EPOCH ==='])
        writer.writerow([*parameter_names, 'epoch', 'seconds', 'loss', 'accuracy'])
        for key, row in per_epoch.items():
            writer.writerow([*parameter_values, *key, *row])

        writer.writerow(['=== BATCH ==='])
        writer.writerow([*parameter_names, 'epoch', 'batch', 'seconds', 'loss'])
        for key, row in per_batch.items():
            writer.writerow([*parameter_values, *key, *row])

    writer.writerow(['=== EVENTS ==='])
    writer.writerow(['id', 'event'])
    for event, id in event_lookup.items():
        writer.writerow([id, event])


def main_timing_stats(infiles, presorted):
    Seconds = NewType('Seconds', float)
    train_batch_times: List[Seconds] = []
    test_times: List[Seconds] = []
    reload_times: List[Seconds] = []
    checkpoint_times: List[Seconds] = []

    def r(times):
        if len(times) == 1:
            return f'random.gauss({mean(times)}, 0)'
        else:
            return f'random.gauss({mean(times)}, {stdev(times)})'

    for tree in Tree.readiter(infiles, presorted=presorted):
        for event in tree.match(Recursive[EventPattern]):
            for child in event.children:
                if len(reload_times) < 1000:
                    for reload in child.match(ReloadPattern):
                        reload_times.append(
                            (reload.times['finished'] - reload.times['started']).total_seconds(),
                        )

                if len(train_batch_times) < 1000:
                    for train in child.match(TrainPattern):
                        for epoch in (epoch for x in train.children for epoch in x.match(EpochPattern)):
                            for batch in (batch for x in epoch.children for batch in x.match(BatchPattern)):
                                train_batch_times.append(
                                    (batch.times['finished'] - batch.times['started']).total_seconds(),
                                )

                if len(test_times) < 1000:
                    for test in child.match(TestPattern):
                        test_times.append(
                            (test.times['finished'] - test.times['started']).total_seconds(),
                        )

                if len(test_times) < 1000:
                    for valid in child.match(ValidPattern):
                        test_times.append(
                            (valid.times['finished'] - valid.times['started']).total_seconds(),
                        )

                if len(checkpoint_times) < 1000:
                    for checkpoint in child.match(CheckpointPattern):
                        checkpoint_times.append(
                            (checkpoint.times['finished'] - checkpoint.times['started']).total_seconds(),
                        )

        if len(reload_times) >= 3 and len(train_batch_times) >= 3 and len(test_times) >= 3 and len(checkpoint_times) >= 3:
            print(f'train batch(n={len(train_batch_times)}): random.gauss({mean(train_batch_times)}, {stdev(train_batch_times)})')
            print(f'test batch(n={len(test_times)}): random.gauss({mean(test_times)}, {stdev(test_times)})')
            print(f'reload(n={len(reload_times)}): random.gauss({mean(reload_times)}, {stdev(reload_times)})')
            print(f'checkpoint(n={len(checkpoint_times)}): random.gauss({mean(checkpoint_times)}, {stdev(checkpoint_times)})')
        else:
            print(f'{len(reload_times) = }')
            print(f'{len(train_batch_times) = }')
            print(f'{len(test_times) = }')
            print(f'{len(checkpoint_times) = }')

        if len(reload_times) >= 1000 and len(train_batch_times) >= 1000 and len(test_times) >= 1000 and len(checkpoint_times) >= 1000:
            break

    print()
    print(f'train batch(n={len(train_batch_times)}): ', end='')
    print(r(train_batch_times))

    print(f'test(n={len(test_times)}): ', end='')
    print(r(test_times))

    print(f'reload(n={len(reload_times)}): ', end='')
    print(r(reload_times))

    print(f'checkpoint(n={len(checkpoint_times)}): ', end='')
    print(r(checkpoint_times))

    def sans_outliers(lst, k=1.5):
        if len(lst) == 1:
            return lst

        # Thanks https://en.wikipedia.org/wiki/Outlier#Tukey's_fences
        q1, q2, q3 = quantiles(lst)
        lo = q1 - k * (q3 - q1)
        hi = q3 + k * (q3 - q1)

        return [x for x in lst if lo <= x <= hi]

    train_batch_times = sans_outliers(train_batch_times)
    test_times = sans_outliers(test_times)
    reload_times = sans_outliers(reload_times)
    checkpoint_times = sans_outliers(checkpoint_times)

    print()
    print(f'train batch(n={len(train_batch_times)}): ', end='')
    print(r(train_batch_times))

    print(f'test(n={len(test_times)}): ', end='')
    print(r(test_times))

    print(f'reload(n={len(reload_times)}): ', end='')
    print(r(reload_times))

    print(f'checkpoint(n={len(checkpoint_times)}): ', end='')
    print(r(checkpoint_times))


def main_debug(infiles, presorted):
    for tree in Tree.readiter(infiles, presorted=presorted):
        tree.pprint()
        print()
        print(tree)
        print()



def cli():
    import argparse

    parser = argparse.ArgumentParser()

    parser.set_defaults(main=None)
    subparsers = parser.add_subparsers(required=True)

    output_csv = subparsers.add_parser('output_csv')
    output_csv.set_defaults(main=main_output_csv)
    output_csv.add_argument('--outfile', '-o', default=sys.stdout, type=argparse.FileType('w'))
    output_csv.add_argument('--presorted', action='store_true')
    output_csv.add_argument('infiles', nargs='+', type=Path)

    timing_stats = subparsers.add_parser('timing_stats')
    timing_stats.set_defaults(main=main_timing_stats)
    timing_stats.add_argument('--presorted', action='store_true')
    timing_stats.add_argument('infiles', nargs='+', type=Path)

    debug = subparsers.add_parser('debug')
    debug.set_defaults(main=main_debug)
    debug.add_argument('--presorted', action='store_true')
    debug.add_argument('infiles', nargs='+', type=Path)

    args = vars(parser.parse_args())
    main = args.pop('main')
    main(**args)


if __name__ == '__main__':
    cli()
