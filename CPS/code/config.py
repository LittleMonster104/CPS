{\rtf1\ansi\ansicpg936\cocoartf2761
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 # config.py\
import os, torch\
from transformers import AutoTokenizer, AutoModel\
\
ENCODER_NAME = "thenlper/gte-large"\
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"\
\
tokenizer = AutoTokenizer.from_pretrained(ENCODER_NAME)\
encoder   = AutoModel.from_pretrained(ENCODER_NAME).to(DEVICE)\
\
def embed(text: str):\
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)\
    with torch.no_grad():\
        outputs = encoder(**inputs)\
    vec = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()\
    return vec / np.linalg.norm(vec)}