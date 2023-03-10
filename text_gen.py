from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import pyarrow as pa
from datetime import datetime
from pathlib import Path
import argparse
import streamlit as st

from example import load
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


st.set_page_config(page_title='Text gen', page_icon='🐱', layout="centered", initial_sidebar_state="collapsed")


@st.cache_resource
def parse_args(args):
    parser = argparse.ArgumentParser('Text gen')
    parser.add_argument('-c', '--ckpt_dir', help='Weights', required=True)    
    parser.add_argument('-t', '--tokenizer_path', help='tokenizer', required=True)
    return parser.parse_args(args)

@st.cache_resource
def load_model(_args):
    return load(_args.ckpt_dir, _args.tokenizer_path, 256, 1)

args = parse_args(sys.argv[1:])
model = load_model(args)

with st.sidebar:
    max_seq_len = st.number_input('max_seq_len', value=256)
    temperature = st.number_input('temperature', value=0.9)
    top_p = st.number_input('top_p', value=0.95)

input_text = st.text_area("Prompt")
b = st.button('Generate', type='primary')
st.subheader("Output")
if b:    
    result_holder = st.empty()
    def progress(p, i, decoded):
        with result_holder.container():
            st.progress(p, f'Progress: Token position={i}')
            if decoded and decoded[0]:
                st.markdown(decoded[0])
    out = model.generate(
        [input_text], max_gen_len=max_seq_len, temperature=temperature, top_p=top_p, callback=progress
    )
    with result_holder.container():
        st.markdown(out[0])
        with st.expander('Raw', expanded=False):
            st.code(out[0], language='markdown')
else:
    st.text('')
