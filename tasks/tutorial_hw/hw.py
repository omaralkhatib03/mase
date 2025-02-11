#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch
torch.manual_seed(0)

from chop.ir.graph.mase_graph import MaseGraph

from chop.passes.graph.analysis import (
    init_metadata_analysis_pass,
    add_common_metadata_analysis_pass,
    add_hardware_metadata_analysis_pass,
    report_node_type_analysis_pass,
)

from chop.passes.graph.transforms import (
    emit_verilog_top_transform_pass,
    emit_internal_rtl_transform_pass,
    emit_bram_transform_pass,
    emit_cocotb_transform_pass,
    quantize_transform_pass,
)

from chop.tools.logger import set_logging_verbosity

set_logging_verbosity("debug")

import toml
import torch
import torch.nn as nn
import os
# os.environ["PATH"] = "/vol/bitbucket/oa321/verilator/verilator/bin:" + os.environ["PATH"]
os.environ["MODULE"] = "top"
get_ipython().system('verilator')


# In[11]:


class MLP(torch.nn.Module):
    """
    Toy FC model for digit recognition on MNIST
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(4, 8)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        x = torch.nn.functional.relu(self.fc1(x))
        return x


# In[12]:


mlp = MLP()
mg = MaseGraph(model=mlp)

# Provide a dummy input for the graph so it can use for tracing
batch_size = 1
x = torch.randn((batch_size, 2, 2))
dummy_in = {"x": x}

mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(
    mg, {"dummy_in": dummy_in, "add_value": False}
)


# In[13]:


config_file = os.path.join(
    os.path.abspath(""),
    "..",
    "..",
    "configs",
    "tests",
    "quantize",
    "fixed.toml",
)
with open(config_file, "r") as f:
    quan_args = toml.load(f)["passes"]["quantize"]
mg, _ = quantize_transform_pass(mg, quan_args)

_ = report_node_type_analysis_pass(mg)

# Update the metadata
for node in mg.fx_graph.nodes:
    for arg, arg_info in node.meta["mase"]["common"]["args"].items():
        if isinstance(arg_info, dict):
            arg_info["type"] = "fixed"
            arg_info["precision"] = [8, 3]
    for result, result_info in node.meta["mase"]["common"]["results"].items():
        if isinstance(result_info, dict):
            result_info["type"] = "fixed"
            result_info["precision"] = [8, 3]


# In[14]:


mg, _ = add_hardware_metadata_analysis_pass(mg)


# In[15]:


from pathlib import  Path

mg, _ = emit_verilog_top_transform_pass(mg)
mg, _ = emit_internal_rtl_transform_pass(mg)


# In[16]:


mg, _ = emit_bram_transform_pass(mg)


# In[17]:


mg, _ = emit_cocotb_transform_pass(mg)


# In[18]:


from chop.actions import simulate

simulate(skip_build=False, skip_test=False)

