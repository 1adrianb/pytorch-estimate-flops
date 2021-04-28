from pthflops.utils import print_table
from .ops_jit import count_ops_jit
try:
    from .ops_fx import count_ops_fx
    force_jit = False
except:
    force_jit = True
    print('Unable to import torch.fx, you pytorch version may be too old.')

__version__ = '0.4.1'


def count_ops(model, input, mode='fx', custom_ops={}, ignore_layers=[], print_readable=True, verbose=True, *args):
    if 'fx' == mode and not force_jit:
        return count_ops_fx(
            model,
            input,
            custom_ops=custom_ops,
            ignore_layers=ignore_layers,
            print_readable=print_table,
            verbose=verbose,
            *args)
    elif 'jit' == mode or force_jit:
        if force_jit:
            print("FX is unsupported on your pytorch version, falling back to JIT")
        return count_ops_jit(
            model,
            input,
            custom_ops=custom_ops,
            ignore_layers=ignore_layers,
            print_readable=print_table,
            verbose=verbose,
            *args)
    else:
        raise ValueError('Unknown mode selected.')
