#!/usr/bin/env python

from importlib import import_module


def monkeypatch(identifier):
    parts = identifier.split('.')
    for i in range(len(parts)):
        print(f'{i = }')
        try:
            print(f'{module_name = }')
            module_name = '.'.join(parts[:-i])
            module = import_module(module_name)
        except ImportError:
            pass
        else:
            break
    else:
        raise ValueError(f'Could not find {identifier}')

    print(f'{module_name = }')
    print(f'{module = }')

    print(f'{parts = }')
    print(f'{parts[i:] = } {i=}')
    parts = parts[i:]
    context = module
    print(f'{context = }')
    for part in parts[:-1]:
        print(f'{part = }')
        context = getattr(current, part)
        print(f'{context = }')

    name = parts[-1]
    previous = getattr(previous, name)

    print(f'{context = }')
    print(f'{name = }')
    print(f'{previous = }')

    def wrapper(func):
        def inner_func(*args, **kwargs):
            new_kwargs = { **kwargs, name: previous }
            return func(*args, **new_kwargs)
        return inner_func
    return wrapper


@monkeypatch('mod.Foo.bar')
def bar(self, s, *, bar):
    print(f'before')
    bar(s)
    print(f'after')


import mod
foo = Foo()
foo.bar('baz')
