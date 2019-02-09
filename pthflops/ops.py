import re
from functools import reduce
from collections import defaultdict
import torch


def string_to_shape(node_string, bias=False):
    r"""Extract the shape of a given tensor from an onnx string

    :param node_string: a :class:`str` or the node from which the shape will be extracted
    :param bias: boolean, if True will return the shape of the bias. If no bias is found the function will return None

    :return: a tuple containing the shape of the tensor
    :rtype: :class:`tuple`
    """
    if not isinstance(node_string, str):
        node_string = str(node_string)
    if bias:
        m = re.search(r"Float\((\d+)\)", node_string)
    else:
        m = re.search(r"Float\(([\d\s\,]+)\)", node_string)
    return m if m is None else tuple(int(x) for x in m.groups()[0].split(','))


def _count_conv2d(node):
    r"""Estimates the number of FLOPs in conv layer

    .. warning::
        Currently it ignore the padding

    :param node_string: an onnx node defining a convolutional layer

    :return: number of FLOPs
    :rtype: `int`
    """
    inp = string_to_shape(list(node.inputs())[0])
    out = string_to_shape(list(node.outputs())[0])
    bias = string_to_shape(list(node.inputs())[0], True)

    f_in = inp[1]
    kh, kw = node['kernel_shape']

    kernel_ops = kh * kw * f_in // node['group']
    bias_ops = 1 if bias is not None else 0
    combined_ops = kernel_ops + bias_ops

    total_ops = combined_ops * reduce(lambda x, y: x * y, out)

    return total_ops


def _count_relu(node):
    r"""Estimates the number of FLOPs of a  ReLU activation.
    The function will count the comparison operation as a FLOP.

    :param node_string: an onnx node defining a ReLU op

    :return: number of FLOPs
    :rtype: `int`
    """
    inp = string_to_shape(list(node.inputs())[0])
    total_ops = 2 * reduce(lambda x, y: x * y, inp)  # also count the comparison
    return total_ops


def _count_avgpool(node):
    r"""Estimates the number of FLOPs of an Average Pooling layer.

    :param node_string: an onnx node defining an average pooling layer

    :return: number of FLOPs
    :rtype: `int`
    """
    out = string_to_shape(list(node.outputs())[0])
    ops_add = reduce(lambda x, y: x * y, node['kernel_shape']) - 1
    ops_div = 1
    total_ops = (ops_add + ops_div) * reduce(lambda x, y: x * y, out)
    return total_ops


def _count_maxpool(node):
    r"""Estimates the number of FLOPs of a Max Pooling layer.

    :param node_string: an onnx node defining a max pooling layer

    :return: number of FLOPs
    :rtype: `int`
    """
    out = string_to_shape(list(node.outputs())[0])
    ops_add = reduce(lambda x, y: x * y, node['kernel_shape']) - 1
    total_ops = ops_add * reduce(lambda x, y: x * y, out)
    return total_ops


def _count_bn(node):
    r"""Estimates the number of FLOPs of a Batch Normalisation operation.

    :param node_string: an onnx node defining a batch norm op

    :return: number of FLOPs
    :rtype: `int`
    """
    inp = string_to_shape(list(node.inputs())[0])
    total_ops = reduce(lambda x, y: x * y, inp) * 2
    return total_ops


def _count_linear(node):
    r"""Estimates the number of a GEMM or linear layer.

    :param node_string: an onnx node defining a GEMM or linear layer

    :return: number of FLOPs
    :rtype: `int`
    """
    inp = string_to_shape(list(node.inputs())[0])
    out = string_to_shape(list(node.outputs())[0])
    f_in = inp[1]
    total_ops = f_in * reduce(lambda x, y: x * y, out)
    return total_ops


def _count_add(node):
    r"""Estimates the number of FLOPs of a summation op.

    :param node_string: an onnx node defining a summation op

    :return: number of FLOPs
    :rtype: `int`
    """
    inp = string_to_shape(list(node.inputs())[0])
    return reduce(lambda x, y: x * y, inp)


def _undefined_op(node):
    r"""Default case for undefined or free (in terms of FLOPs) operations

    :param node_string: an onnx node

    :return: always 0
    :rtype: `int`
    """
    return 0

count_operations = defaultdict(
    lambda: _undefined_op,
    {
        'onnx::Conv': _count_conv2d,
        'onnx::Relu': _count_relu,
        'onnx::AveragePool': _count_avgpool,
        'onnx::MaxPool': _count_maxpool,
        'onnx::BatchNormalization': _count_bn,
        'onnx::Gemm': _count_linear,
        'onnx::Add': _count_add
    }
)


def count_ops(model, input, print_readable=True, *args):
    r"""Estimates the number of FLOPs of an :class:`torch.nn.Module`

    :param model: the :class:`torch.nn.Module`
    :param input: a N-d :class:`torch.tensor` containing the input to the model
    :param print_readable: boolean, if True will print the number of FLOPs. default is True

    :return: number of FLOPs
    :rtype: `int`
    """
    # Make sure that the input is on the same device as the model
    if next(model.parameters()).device != input.device:
        input.to(next(model.parameters()).device)

    # Convert pytorch module to ONNX
    trace, _ = torch.jit.get_trace_graph(model, input, *args)
    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)
    graph = trace.graph()

    ops = 0
    for node in graph.nodes():
        ops += count_operations[node.kind()](node)

    if print_readable:
        print("Input size: {0}".format(tuple(input.shape)))
        print("{:,} FLOPs or approx. {:,.2f} GFLOPs".format(ops, ops / 1e+9))

    return ops
