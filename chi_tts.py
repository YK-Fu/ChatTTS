import torch
import librosa
import soundfile as sf
import json
import os
from tqdm import tqdm
import soundfile
import ChatTTS
import torch
import torchaudio
import opencc
from tn.chinese.normalizer import Normalizer as ZhNormalizer
from glob import glob
def generate(line):
    try:
        diag = json.loads(line)
        name = diag['id']
        os.makedirs(f"./chi_audio/{name}", exist_ok=True)
    except:
        return
    for role in ["Machine", "User"]:
        idx = 1
        texts = []
        ids = []
        rand_spk = chat_model.sample_random_speaker()
        params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb = rand_spk, # add sampled speaker 
            temperature = .3,
            top_P = 0.7,
            top_K = 20, 
        )
        while True:
            key = f"{role}_{idx}"
            if key in diag:
                ids.append(key)
                text = zh_tn_model.normalize(converter.convert(diag[key].replace('？', '，').replace('！', '，').replace('。', '，').replace('\n', ' ').replace('  ', ' ').replace('-', ' ')))
                if text:
                    texts.append(text)
                    idx += 1
                else:
                    break
            else:
                break
        if len(glob(f'chi_audio/{name}/{role}_*.wav')) == len(texts):
            continue
        with torch.no_grad():
            audios = chat_model.infer(texts,  params_infer_code=params_infer_code)
            
        for key, audio, text in zip(ids, audios, texts):
            audio = librosa.resample(audio, orig_sr=24000, target_sr=16000)
            sf.write(f'chi_audio/{name}/{key}.wav', audio.T, 16000)
            with open(f'chi_audio/{name}/{key}.txt', 'w') as txt:
                txt.write(text)
                txt.write('\n')


converter = opencc.OpenCC('t2s.json')
chat = ChatTTS.Chat()
chat.load(compile=False) # Set to True for better performance

import sys
file = sys.argv[1]
cuda_idx = int(sys.argv[2])
chat_model = ChatTTS.Chat()
chat_model.load(compile=False, device=torch.device(f"cuda:{cuda_idx}"))
zh_tn_model = ZhNormalizer(remove_erhua=True, overwrite_cache=True)
with open(file, encoding='utf-8', errors='ignore') as f:
    for line in tqdm(f.readlines()):
        generate(line)