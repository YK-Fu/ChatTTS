import subprocess
import os
import json
from glob import glob
if __name__ == '__main__':
    batch = 32
    N_GPUS = 8
    file_list = []
    with open("/mnt/home/ivenfu/dialogue/dialogue_chinese_llama31_70B_diverse_new.jsonl", encoding='utf-8', errors='ignore') as f:
        for i, line in enumerate(f.readlines()):
            data = json.loads(line.strip())
            data['id'] = i
            file_list.append(json.dumps(data, ensure_ascii=False))

    os.makedirs('./tmp', exist_ok=True)
    temp_files = [open(f'./tmp/{i}.jsonl', 'w') for i in range(batch)]
    for i, file in enumerate(file_list):
        out = temp_files[i % batch]
        out.write(file)
        out.write('\n')
    [f.close() for f in temp_files]
    for i in range(batch):
        cmd = 'python chi_tts.py {} {}'.format(f'./tmp/{i}.jsonl', i % N_GPUS)
        cmd_process = subprocess.Popen([cmd], shell=True)