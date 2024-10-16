
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
import pickle

from torch.fx import Interpreter

import torch.fx
import json



graph_idx=0

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




def check_nan_inf_plugin():
    pass

def global_plugin_pass():
    pass


class TraceInterpreter(Interpreter):
    def __init__(self, module):
        super().__init__(module)
        self.op_trace = {}  # 用于存储每个操作的输入和输出
        self.datas = {}
    def run_node(self, n):
        input_names = [i.name if isinstance(i, torch.fx.Node) else i for i in n.args]
        result = super().run_node(n)
        node_name = n.name
        res = {}
        res_name = []
        if isinstance(result, tuple) or isinstance(result, list):
            res = { node_name + f".res{idx}" : value.float().numpy() for idx, value in enumerate(result) }
            res_name = [node_name + f".res{idx}" for idx in range(len(result))]
        else:
            idx = 0
            res[ node_name + f".res{idx}"] = result.float().numpy()
            res_name = [node_name + f".res{idx}"]
        # self.op_trace.append({
        #     'node_name': n.name,
        #     'op': node_name,
        #     'inputs': input_names,
        #     'outputs': res,
        # })
        self.op_trace[n.name] = {"inputs": input_names, "outputs": res_name}
        if "getitem" not in node_name:
            self.datas.update(res)
        return result

# def warp_calc(module, idx=0):
#     inner_idx = idx
#     def forward(*args):
#         tinputs = args
#         res = module(tinputs)
#         return res
#     return forward


def save_2_pickle(fx_g, path):
    with open(path, 'wb') as f:
        pickle.dump(fx_g, f)

global_run_idx = 0

def warp_calc(module, idx=0):
    # 使用 TraceInterpreter 来执行并跟踪每个节点
    tracer = TraceInterpreter(module)
    
    def forward(*args):
        global global_run_idx
        # 运行整个图并跟踪
        res = tracer.run(*args)
        if global_run_idx < 2:
            # save all data 
            import pdb;pdb.set_trace()
            json.dump(tracer.op_trace, open(f"trace_data_{global_run_idx}.json", "w"))
            np.savez(f"trace_data_{global_run_idx}.npz", **tracer.datas)
        global_run_idx += 1
        return res

    return forward

def tpu_mlir_compiler(fx_g, example_inputs):
    global graph_idx
    time_str = f'{graph_idx}'
    graph_idx += 1
    os.system(f'mkdir -p base{time_str}')
    fx_g.to_folder(f'fx_graph_dumped_{time_str}', "test")
    # torch.save(fx_g, f'fx_graph_dumped_{time_str}.pth')
    # fx_g.to_pickle_file(f'fx_graph_dumped_{time_str}.pkl')
    save_2_pickle(fx_g, f'fx_graph_dumped_{time_str}.pkl')
    print([list(i.shape) for i in example_inputs])
    # return warp_calc(make_boxed_func(fx_g.forward), graph_idx)
    return warp_calc(fx_g)

aot_backend = aot_autograd(fw_compiler = tpu_mlir_compiler, bw_compiler = tpu_mlir_compiler, decompositions=_get_disc_decomp())
