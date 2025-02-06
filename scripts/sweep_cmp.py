import json
import os
import subprocess
import re
from tqdm import tqdm

import numpy as np

# cmd = "python3 -m llm_analysis.analysis infer -m meta-llama/Llama-3.1-405b -g custom -t 12 -p 2 -b 128 --seq_len 2000 -n 200 --log_level DEBUG"
gpu_config_file = "llm_analysis/gpu_configs/custom.json"
# flops_name = "peak_fp16_TFLOPS"
mem_bw_name = "hbm_bandwidth_in_GB_per_sec"
mem_cap_name = "mem_per_GPU_in_GB"

# os.chdir("..")
num_gpus = np.arange(2, 26, 2)
num_mems = np.arange(1, 25, 1)
mem_power_function = lambda x: 150 + 50 * x ** 2
# print(num_mems)
tps = {
    "prefill": {},
    "decode": {}
}

lat = {
    "prefill": {},
    "decode": {}
}

power = {}

with tqdm(range(len(num_gpus) * len(num_mems))) as pbar:
    for gpus in num_gpus:
        for _, v in tps.items():
            v[gpus] = {}
        for _, v in lat.items():
            v[gpus] = {}
        power[gpus] = {}

        for mems in num_mems:
            with open(gpu_config_file, 'r') as f:
                gpu_config_str = f.read()
                gpu_config_dict = json.loads(gpu_config_str)
                gpu_config_dict[mem_bw_name] = float(mems * 7200 / gpus)
                gpu_config_dict[mem_cap_name] = float(mems * 216 / gpus)
            with open(gpu_config_file, 'w') as f:
                f.write(json.dumps(gpu_config_dict))
            tp = int(gpus)
            pp = 1
            cmd = f"python3 -m llm_analysis.analysis infer -m meta-llama/Llama-3.1-405b -g custom -t {tp} -p {pp} -b 16 --seq_len 200 -n 100000 --log_level DEBUG --calculate_gpu_power"
            try:
                out = subprocess.check_output(cmd.split(' '), cwd=".", stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError as e:
                # print(e.output)
                pbar.update(1)
                continue
            for r in [r'"prefill_tokens_per_sec": (.*?),', r'"decode_tokens_per_sec": (.*?),']:
                match = re.findall(r, str(out))
                tps_type = "prefill" if "prefill" in r else "decode"
                tps[tps_type][gpus][mems] = float(match[0])

            r = r'latency_per_layer: (.*?) ms'
            match = re.findall(r, str(out))
            lat["prefill"][gpus][mems] = float(match[0])
            lat["decode"][gpus][mems] = float(match[1])

            r = r'"total_power": (.*?),'
            match = re.findall(r, str(out))
            gpu_power = float(match[0])

            r = r'"memory_utilization": (.*?),'
            match = re.findall(r, str(out))
            mem_util = float(match[0])

            power[gpus][mems] = gpu_power + mems * mem_power_function(mem_util)

            pbar.update(1)

print(tps)
print(lat)
print(power)