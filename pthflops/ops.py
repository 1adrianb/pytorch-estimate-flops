import re
from functools import reduce
from collections import defaultdict
from distutils.version import LooseVersion
from typing import Iterable
import torch

from .utils import print_table, scope_name_workaround


def string_to_shape(node_string, bias=False):
    r"""Extract the shape of a given tensor from an onnx string

    :param node_string: a :class:`str` or the node from which the shape will be extracted
    :param bias: boolean, if True will return the shape of the bias. If no bias is found the function will return None

    :return: a tuple containing the shape of the tensor
    :rtype: :class:`tuple`
    """
    if not isinstance(node_string, str):
        node_string = str(node_string)
    node_string = node_string.replace('!', '')
    if bias:
        m = re.search(r"Float\((\d+)\)", node_string)
    else:
        m = re.search(r"Float\(([\d\s\,]+)\)", node_string)
    return m if m is None else tuple(int(x) for x in m.groups()[0].split(','))


def _parse_node_inputs(node, version=2):
    inputs = {}
    inputs_names = []
    for idx, inp in enumerate(node.inputs()):
        inp = str(inp)
        curr_node_name = re.search(r'(.*) defined in ', inp).group(1)
        if version <= 2:
            extracted_data = re.search(r'%' + curr_node_name + r' : Float\(([^%]*)\)[,| ]', inp)
        elif version == 3:
            extracted_data = re.search(r'%' + curr_node_name + r' : Float\(((\d+, )+)', inp)

        if extracted_data is not None:
            extracted_data = extracted_data.group(1)
        else:
            return _parse_node_inputs(
                list(node.inputs())[0].node()
            )
        if version <= 2:
            inputs[curr_node_name] = re.findall(r'(\d+):', extracted_data)
        elif version == 3:
            inputs[curr_node_name] = re.findall(r'(\d+)', extracted_data)
        inputs[curr_node_name] = list(map(int, inputs[curr_node_name]))
        inputs_names.append(curr_node_name)
    return inputs, inputs_names


def parse_node_info(node, version=2):
    inputs, inputs_names = _parse_node_inputs(node, version=version)
    node = str(node)
    node_name = re.search(r'%(.*) : ', node).group(1)
    if version == 2:
        out_size = re.search(r'Float\(\d+:(\d+),', node).group(1)
    elif version == 3:
        out_size = re.search(r'strides=\[(\d+),', node).group(1)

    return node_name, inputs, inputs_names, int(out_size)


def _count_convNd(node, version=2):
    r"""Estimates the number of FLOPs in conv layer

    .. warning::
        Currently it ignore the padding

    :param node_string: an onnx node defining a convolutional layer

    :return: number of FLOPs
    :rtype: `int`
    """
    kernel_size = node['kernel_shape']
    num_groups = node['group']

    if version == 1:
        inp = string_to_shape(list(node.inputs())[0])
        out = string_to_shape(list(node.outputs())[0])
        out_ops = reduce(lambda x, y: x * y, out)
        bias_ops = 1 if string_to_shape(list(node.inputs())[0], True) is not None else 0
        f_in = inp[1]
    elif version in [2, 3]:
        node_name, inputs, inputs_names, out_ops = parse_node_info(node, version=version)
        f_in = inputs[inputs_names[0]][1]
        bias_ops = 1 if len(inputs_names) == 3 else 0

    kernel_ops = f_in
    for ks in kernel_size:
        kernel_ops *= ks

    kernel_ops = kernel_ops // num_groups
    combined_ops = kernel_ops + bias_ops

    total_ops = combined_ops * out_ops

    return total_ops


def _count_relu(node, version=2):
    r"""Estimates the number of FLOPs of a  ReLU activation.
    The function will count the comparison operation as a FLOP.

    :param node_string: an onnx node defining a ReLU op

    :return: number of FLOPs
    :rtype: `int`
    """
    if version == 1:
        inp = string_to_shape(list(node.inputs())[0])
    elif version in [2, 3]:
        node_name, inputs, inputs_names, out_ops = parse_node_info(node, version=version)
        inp = inputs[inputs_names[0]]
    total_ops = 2 * reduce(lambda x, y: x * y, inp)  # also count the comparison
    return total_ops


def _count_avgpool(node, version=2):
    r"""Estimates the number of FLOPs of an Average Pooling layer.

    :param node_string: an onnx node defining an average pooling layer

    :return: number of FLOPs
    :rtype: `int`
    """
    if version == 1:
        out = string_to_shape(list(node.outputs())[0])
        out_ops = reduce(lambda x, y: x * y, out)
    elif version in [2, 3]:
        node_name, inputs, inputs_names, out_ops = parse_node_info(node, version=version)

    ops_add = reduce(lambda x, y: x * y, node['kernel_shape']) - 1
    ops_div = 1
    total_ops = (ops_add + ops_div) * out_ops
    return total_ops


def _count_globalavgpool(node, version=2):
    r"""Estimates the number of FLOPs of an Average Pooling layer.

    :param node_string: an onnx node defining an average pooling layer

    :return: number of FLOPs
    :rtype: `int`
    """
    if version == 1:
        inp = string_to_shape(list(node.inputs())[0])
        out = string_to_shape(list(node.outputs())[0])
        out_ops = reduce(lambda x, y: x * y, out)
    elif version in [2, 3]:
        node_name, inputs, inputs_names, out_ops = parse_node_info(node, version=version)
        inp = inputs[inputs_names[0]]

    ops_add = reduce(lambda x, y: x * y, [inp[-2], inp[-1]]) - 1
    ops_div = 1
    total_ops = (ops_add + ops_div) * out_ops
    return total_ops


def _count_maxpool(node, version=2):
    r"""Estimates the number of FLOPs of a Max Pooling layer.

    :param node_string: an onnx node defining a max pooling layer

    :return: number of FLOPs
    :rtype: `int`
    """
    if version == 1:
        out = string_to_shape(list(node.outputs())[0])
        out_ops = reduce(lambda x, y: x * y, out)
    elif version in [2, 3]:
        node_name, inputs, inputs_names, out_ops = parse_node_info(node, version=version)

    ops_add = reduce(lambda x, y: x * y, node['kernel_shape']) - 1
    total_ops = ops_add * out_ops
    return total_ops


def _count_bn(node, version=2):
    r"""Estimates the number of FLOPs of a Batch Normalisation operation.

    :param node_string: an onnx node defining a batch norm op

    :return: number of FLOPs
    :rtype: `int`
    """
    if version == 1:
        if 'BatchNorm1d' in node.scopeName():
            inp = string_to_shape(list(node.inputs())[1])
        else:
            inp = string_to_shape(list(node.inputs())[0])
    elif version in [2, 3]:
        node_name, inputs, inputs_names, out_ops = parse_node_info(node, version=version)
        inp = inputs[inputs_names[0]]

    total_ops = reduce(lambda x, y: x * y, inp) * 2
    return total_ops


def _count_linear(node, version=2):
    r"""Estimates the number of a GEMM or linear layer.

    :param node_string: an onnx node defining a GEMM or linear layer

    :return: number of FLOPs
    :rtype: `int`
    """
    if version == 1:
        inp = string_to_shape(list(node.inputs())[0])
        out = string_to_shape(list(node.outputs())[0])
        f_in = inp[1]
        out_ops = reduce(lambda x, y: x * y, out)
    elif version in [2, 3]:
        node_name, inputs, inputs_names, out_ops = parse_node_info(node, version=version)
        inp = inputs[inputs_names[0]]
        f_in = inputs[inputs_names[0]][1]

    total_ops = f_in * out_ops
    return total_ops


def _count_add_mul(node, version=2):
    r"""Estimates the number of FLOPs of a summation op.

    :param node_string: an onnx node defining a summation op

    :return: number of FLOPs
    :rtype: `int`
    """
    if version == 1:
        inp = string_to_shape(list(node.inputs())[0])
    elif version in [2, 3]:
        node_name, inputs, inputs_names, out_ops = parse_node_info(node, version=version)
        inp = inputs[inputs_names[0]]
    return reduce(lambda x, y: x * y, inp)


def _undefined_op(node, version=2):
    r"""Default case for undefined or free (in terms of FLOPs) operations

    :param node_string: an onnx node

    :return: always 0
    :rtype: `int`
    """
    return 0

count_operations = defaultdict(
    lambda: _undefined_op,
    {
        'onnx::Conv': _count_convNd,
        'onnx::Relu': _count_relu,
        'onnx::AveragePool': _count_avgpool,
        'onnx::MaxPool': _count_maxpool,
        'onnx::BatchNormalization': _count_bn,
        'onnx::Gemm': _count_linear,
        'onnx::MatMul': _count_linear,
        'onnx::Add': _count_add_mul,
        'onnx::Mul': _count_add_mul,
        'onnx::GlobalAveragePool': _count_globalavgpool
    }
)


def count_ops(model, input, custom_ops={}, ignore_layers=[], print_readable=True, verbose=True, *args):
    r"""Estimates the number of FLOPs of an :class:`torch.nn.Module`

    :param model: the :class:`torch.nn.Module`
    :param input: a N-d :class:`torch.tensor` containing the input to the model
    :param custom_ops: :class:`dict` containing custom counting functions. The keys represent the name
    of the targeted aten op, while the value a lambda or callback to a function returning the number of ops.
    This can override the ops present in the package.
    :param ignore_layers: :class:`list` containing the name of the modules to be ignored.
    :param print_readable: boolean, if True will print the number of FLOPs. default is True
    :param verbose: boolean, if True will print all the non-zero OPS operations from the network

    :return: number of FLOPs
    :rtype: `int`
    """
    # Remove dataparallel wrapper if present
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    # Make sure that the input is on the same device as the model
    input_device = input.device if not isinstance(input, Iterable) else input[0].device
    if next(model.parameters()).device != input_device:
        if isinstance(input, Iterable):
            for inp in input:
                inp.to(next(model.parameters()).device)
        else:
            input.to(next(model.parameters()).device)

    # Place the model in eval mode, required for some models
    model_status = model.training
    model.eval()

    # Convert pytorch module to ONNX
    version = 1
    if LooseVersion(torch.__version__) > LooseVersion('1.3.1'):
        with scope_name_workaround():
            trace, _ = torch.jit._get_trace_graph(model, input, *args)
            graph = torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

        if LooseVersion(torch.__version__) >= LooseVersion('1.6.0') and \
                LooseVersion(torch.__version__) < LooseVersion('1.8.0'):
            version = 2
        elif LooseVersion(torch.__version__) >= LooseVersion('1.8.0'):
            version = 3
    else:
        # PyTorch 1.3 and bellow
        trace, _ = torch.jit.get_trace_graph(model, input, *args)
        torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
        graph = trace.graph()

    ops = 0
    all_data = []
    for node in graph.nodes():
        if any(name in node.scopeName() for name in ignore_layers):
            continue
        if node.kind() in custom_ops.keys():
            current_ops = custom_ops[node.kind()](node, version)
        else:
            current_ops = count_operations[node.kind()](node, version)
        ops += current_ops

        if current_ops and verbose:
            all_data.append(['{}/{}'.format(node.scopeName(), node.kind()), current_ops])

    if print_readable:
        if verbose:
            print_table(all_data)
        print("Input size: {0}".format(tuple(input.shape)))
        print("{:,} FLOPs or approx. {:,.2f} GFLOPs".format(ops, ops / 1e+9))

    if model_status:
        model.train()

    return ops, all_data
