
# -*- coding: utf-8 -*-
import os
import torch
import time
import copy
from argparse import Namespace
from torch._dynamo.backends.common import aot_autograd
from functorch.compile import make_boxed_func
from torch._functorch import compilers
# from fx_pass import fx_pass_for_bmm_expand
import pdb
from torch.fx.experimental.proxy_tensor import maybe_disable_fake_tensor_mode
import numpy as np
import gc
import subprocess
import importlib
import json
graph_idx=0


from torch.fx import Interpreter

import torch
import torch.fx
import json

def _get_disc_decomp():
    from torch._decomp import get_decompositions
    aten = torch.ops.aten
    decompositions_dict = get_decompositions(
        [
            # aten.var_mean,
            # aten._adaptive_avg_pool2d_backward,
            # aten.addcmul,
            # aten.avg_pool2d_backward,
            # aten.binary_cross_entropy_with_logits,
            aten.gelu,
            aten.gelu_backward,
            # aten.glu_backward,
            # aten.grid_sampler_2d,
            # aten.hardsigmoid,
            # aten.hardsigmoid_backward,
            # aten.hardswish,
            # aten.hardswish_backward,
            # aten.hardtanh,
            # aten.hardtanh_backward,
            # aten.logsumexp.default,
            # aten.max_pool2d_with_indices_backward,
            # aten.mse_loss,
            # aten.mse_loss_backward,
            # aten.mv,
            # aten.narrow,
            # aten.native_batch_norm,
            # aten.native_batch_norm_backward,
            # aten.native_dropout_backward,
            # aten.native_group_norm,
            aten.native_group_norm_backward,
            # aten.native_layer_norm,
            aten.native_layer_norm_backward,
            # aten.std_mean.correction,
            # aten._softmax,
            aten._softmax_backward_data,
            # aten.stack,
            # aten.t,
            aten.tanh_backward,
            aten.slice_backward,
            aten.convolution_backward,
            aten.select_backward,
            aten.embedding_dense_backward,
            aten.sigmoid_backward,
            aten.nll_loss_backward,
            aten._log_softmax_backward_data,
            aten.nll_loss_forward,
            aten.mse_loss,
            aten.mse_loss_backward,
            aten._scaled_dot_product_flash_attention.default
        ]
    )
    return decompositions_dict

from torch.fx import Interpreter


class TraceInterpreter(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.op_trace = []  # 用于存储每个操作的输入和输出
    
    def run_node(self, n):
        # 获取输入，直接使用 n.args 获取传入的参数
        inputs = [self.fetch_attr(i) if isinstance(i, str) else i for i in n.args]

        # 记录输入的形状或值
        input_shapes = []
        for i in inputs:
            if isinstance(i, torch.Tensor):
                input_shapes.append(tuple(i.shape))  # 获取张量的形状
            else:
                import pdb;pdb.set_trace()
                input_shapes.append(str(i))  # 对于非张量，记录其值
        
        # 执行当前 node 的操作
        result = super().run_node(n)
        
        # 记录输出的形状或值
        output_shapes = []
        if isinstance(result, tuple):
            for r in result:
                if isinstance(r, torch.Tensor):
                    output_shapes.append(tuple(r.shape))
                else:
                    import pdb;pdb.set_trace()
                    output_shapes.append(str(r))
        elif isinstance(result, torch.Tensor):
            output_shapes.append(tuple(result.shape))
        else:
            import pdb;pdb.set_trace()
            output_shapes.append(str(result))
        
        # 记录输入和输出
        self.op_trace.append({
            'node_name': n.name,
            'op': n.target,
            'inputs': input_shapes,
            'outputs': output_shapes,
        })
        
        print(f"Node {n.name}: Inputs: {input_shapes}, Outputs: {output_shapes}")
        return result

# def warp_calc(module, idx=0):
#     inner_idx = idx
#     def forward(*args):
#         tinputs = args
#         res = module(tinputs)
#         return res
#     return forward

def warp_calc(module, idx=0):
    # 使用 TraceInterpreter 来执行并跟踪每个节点
    tracer = TraceInterpreter(module)
    
    def forward(*args):
        # 运行整个图并跟踪
        res = tracer.run(*args)
        return res
    
    return forward

def tpu_mlir_compiler(fx_g, example_inputs):
    global graph_idx
    time_str = f'{graph_idx}'
    graph_idx += 1
    os.system(f'mkdir -p base{time_str}')
    fx_g.to_folder(f'fx_graph_dumped_{time_str}', "test")
    print([list(i.shape) for i in example_inputs])
    # return warp_calc(make_boxed_func(fx_g.forward), graph_idx)
    return warp_calc(fx_g)

aot_backend = aot_autograd(fw_compiler = tpu_mlir_compiler, bw_compiler = tpu_mlir_compiler, decompositions=_get_disc_decomp())
