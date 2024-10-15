
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


def warp_calc(module, idx=0):
    inner_idx = idx
    def forward(*args):
        print(">>>>>>> inputs ")
        for i in range(len(args)):
            print(args[i].abs().max(), args[i].abs().min())
        tinputs = [i.half() if i.dtype == torch.float32 else i for i in args]
        res = module(tinputs)
        print(">>>>>>> outputs ")
        for i in range(len(res)):
            if res[i] is not None:
                print(res[i].abs().max(), res[i].abs().min())
        return res
    return forward

def tpu_mlir_compiler(fx_g, example_inputs):
    global graph_idx
    time_str = f'{graph_idx}'
    graph_idx += 1
    os.system(f'mkdir -p base{time_str}')
    fx_g.to_folder(f'fx_graph_dumped_{time_str}', "test")
    print([list(i.shape) for i in example_inputs])
    return warp_calc(make_boxed_func(fx_g.forward), graph_idx)

aot_backend = aot_autograd(fw_compiler = tpu_mlir_compiler, bw_compiler = tpu_mlir_compiler, decompositions=_get_disc_decomp())#fw_compiler=skip_compiler,
