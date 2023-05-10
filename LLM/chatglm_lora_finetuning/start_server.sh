#!/bin/bash
cd /root/autodl-tmp/chatglm/my_chatglm_lora 
nohup /root/miniconda3/bin/gunicorn flask_stream_server_v3:app -b 0.0.0.0:6006 -w 1 -t 600 > ./log.log 2>&1 & 
