#!/usr/bin/env python

from importlib import import_module
import logging

logger = logging.getLogger(__name__)


def monkeypatch(identifier):
    parts = identifier.split('.')
    for i in range(1, len(parts)):
        logging.debug(f'{i = }')
        try:
            module_name = '.'.join(parts[:-i])
            logger.debug(f'{module_name = }')
            module = import_module(module_name)
        except ImportError:
            pass
        else:
            break
    else:
        raise ValueError(f'Could not find {identifier}')

    logger.debug(f'{module_name = }')
    logger.debug(f'{module = }')

    logger.debug(f'{parts = }')
    logger.debug(f'{parts[-i:] = } {i=}')
    parts = parts[-i:]
    context = module
    logger.debug(f'{context = }')
    for part in parts[:-1]:
        logger.debug(f'{part = }')
        context = getattr(context, part)
        logger.debug(f'{context = }')

    name = parts[-1]
    previous = getattr(context, name)

    logger.debug(f'{context = }')
    logger.debug(f'{name = }')
    logger.debug(f'{previous = }')

    def wrapper(func):
        def inner_func(*args, **kwargs):
            new_kwargs = { **kwargs, name: previous }
            return func(*args, **new_kwargs)
        setattr(context, name, inner_func)
        return inner_func
    return wrapper
