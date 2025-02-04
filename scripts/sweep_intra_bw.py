import json
import os
import subprocess
import re
from tqdm import tqdm

import numpy as np

cmd = "python -m llm_analysis.analysis infer -m meta-llama/Llama-3.1-405b -g custom -t 12 -p 2 -b 128 --seq_len 2000 -n 200 --log_level DEBUG"
gpu_config_file = "llm_analysis/gpu_configs/custom.json"
flops_name = "peak_fp16_TFLOPS"
mem_bw_name = "hbm_bandwidth_in_GB_per_sec"

os.chdir("..")
flops_range = np.arange(100, 2100, 200)
mem_bw_range = np.arange(1200, 8400, 800)

# tps = {
#     "prefill": {},
#     "decode": {}
# }
#
# with tqdm(range(len(flops_range) * len(mem_bw_range))) as pbar:
#     for flops in flops_range:
#         for _, v in tps.items():
#             v[flops] = {}
#         for mem_bw in mem_bw_range:
#             with open(gpu_config_file, 'r') as f:
#                 gpu_config_str = f.read()
#                 gpu_config_dict = json.loads(gpu_config_str)
#                 gpu_config_dict[flops_name] = float(flops)
#                 gpu_config_dict[mem_bw_name] = float(mem_bw)
#             with open(gpu_config_file, 'w') as f:
#                 f.write(json.dumps(gpu_config_dict))
#             out = subprocess.check_output(cmd.split(' '), cwd=".", stderr=subprocess.STDOUT)
#             for r in [r'"prefill_tokens_per_sec": (.*?),', r'"decode_tokens_per_sec": (.*?),']:
#                 match = re.findall(r, str(out))
#                 tps_type = "prefill" if "prefill" in r else "decode"
#                 if len(match) > 1:
#                     tps[tps_type][flops][mem_bw] = float(match[0])
#             pbar.update(1)
#
# print(tps)



# factor = [1, 2, 3, 4, 6, 8, 12, 24]
# r = r'"prefill_tokens_per_sec": (.*?),'
# r = r'latency_per_layer: (.*?) ms'
# for tp in factor:
#     cmd = f"python -m llm_analysis.analysis infer -m meta-llama/Llama-3.1-405b -g h200-sxm-141gb -t {tp} -p {24//tp} -b 128 --seq_len 2000 -n 200 --log_level DEBUG"
#     # print(cmd)
#     out = subprocess.check_output(cmd.split(' '), cwd="..", stderr=subprocess.STDOUT)
#     # print(out)
#     match = re.findall(r, str(out))
#     if match is not None:
#         print(f"{match[1]},")

batch_sizes = list(map(lambda x: 2**x, [i for i in range(11)]))
parallelism_combo = [
    (i, 24 // i) for i in range(1, 25) if 24 % i == 0
]

data = {}
with tqdm(range(len(parallelism_combo) * len(batch_sizes))) as pbar:
    for tp, pp in parallelism_combo:
        data[f"{tp}_{pp}"] = {
            "lat": {
                "prefill": [],
                "decode": [],
            },
            "th": {
                "prefill": [],
                "decode": [],
            }
        }
        for batch_size in batch_sizes:
            cmd = f"python -m llm_analysis.analysis infer -m meta-llama/Llama-3.1-405b -g custom -t {tp} -p {pp} -b {batch_size} --seq_len 2000 -n 200 --log_level DEBUG"
            try:
                out = subprocess.check_output(cmd.split(' '), stderr=subprocess.STDOUT)
                for r in [r'"prefill_tokens_per_sec": (.*?),', r'"decode_tokens_per_sec": (.*?),']:
                    match = re.findall(r, str(out))
                    tps_type = "prefill" if "prefill" in r else "decode"
                    if len(match) > 0:
                        data[f"{tp}_{pp}"]["th"][tps_type].append(float(match[0]))
                    else:
                        data[f"{tp}_{pp}"]["th"][tps_type].append(0)
                match = re.findall(r"latency_per_layer: (.*?) ms", str(out))
                if len(match) >= 2:
                    data[f"{tp}_{pp}"]["lat"]["prefill"].append(float(match[0]))
                    data[f"{tp}_{pp}"]["lat"]["decode"].append(float(match[1]))
                pbar.update(1)
            except subprocess.CalledProcessError:
                pbar.update(1)
                continue

print(data)

