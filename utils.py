#!/usr/bin/env python

from functools import reduce, partial
from itertools import islice
from collections import deque
from multiprocessing import Pool, cpu_count
from typing import Iterable, Callable

def chunks(data, SIZE=1000):
    if isinstance(data, Iterable) or isinstance(data, deque):
        it = iter(data)
        for i in range(0, len(data), SIZE):
            if isinstance(data, dict):
                yield {k: data[k] for k in islice(it, SIZE)}
            else:
                yield [k for k in islice(it, SIZE)]

def parallel_map(func, args = (),
                 merge_func = partial(reduce, lambda a, b: a + b),
                 cores = cpu_count(),
                 CHUNKSIZE = 1000):
    def decorator(func: Callable):
        def inner():
            with Pool(cores) as p:
                results = p.map(func, chunks(args, CHUNKSIZE))
            return merge_func(results)
        return inner
    return decorator(func)

def flatten(list_of_items):
    ''' Flattens list of lists exactly 1 level '''
    return [num for sublist in list_of_items for num in sublist]

def dummy_iterator(chunked: Iterable, iterfunc = Callable, **kwargs):
    results = [] # Change to deque?
    if isinstance(chunked, dict):
        for k, v in chunked.items():
            results.append(iterfunc(k, v))
    else:
        for i in chunked:
            results.append(iterfunc(i))
    return results

def parallelize(func = Callable, args = Iterable, CHUNKSIZE = 1000,
                 cores = cpu_count(),
                 merge_func = partial(reduce, lambda a, b: a + b),
                 compress = False, **kwargs):
    ''' Routine for parallelizing function with a single iterable argument

        For complex functions, write a wrapper that parses out the iterable
        and sends the relevant parameters to the function.

        CHUNKSIZE: int (size of the blocks of an iterable over which
                        you want to parallelize)


        Example: To fit data to a curve using scipy's curve_fit, create
                 a wrapper function as follows

        >> def fit_wrapper(arg, fitfunc = Callable, **kwargs):
        >>     args, pcov = curve_fit(fitfunc, arg['xcolumn'], arg['ycolumn'], **kwargs)
        >>     return args, pcov

        >> f = parallelize(partial(fit_data, fitfunc = myfittingfunc,
        >>                           maxfev = 400),
        >>                   iterable_args, CHUNKSIZE = 100)

        Alternatively, you can simply provide kwargs to the parallelize function:
        (Actually, check that this is true before using!)
        >> f = parallelize(fit_data, iterable_args, fitfunc = myfittingfunc,
        >>                 **fitting_function_kwargs)
    '''
    def inner():
        with Pool(cores) as p:
            results = p.map(partial(dummy_iterator, iterfunc = func, **kwargs), chunks(args, CHUNKSIZE))
        return merge_func(results)
    if compress:
        return list(filter(None, inner()))
    else:
        return inner()

run_parallel = parallelize
