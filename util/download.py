import os
from tqdm import tqdm
import requests


def download_pretrained_vae(overwrite=False):
    download_path = "pretrained_models/vae/kl16.ckpt"
    if not os.path.exists(download_path) or overwrite:
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        os.makedirs("pretrained_models/vae", exist_ok=True)
        r = requests.get("https://www.dropbox.com/scl/fi/hhmuvaiacrarfg28qxhwz/kl16.ckpt?rlkey=l44xipsezc8atcffdp4q7mwmh&dl=0", stream=True, headers=headers)
        print("Downloading KL-16 VAE...")
        with open(download_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=254):
                if chunk:
                    f.write(chunk)


def download_pretrained_marb(overwrite=False):
    download_path = "pretrained_models/mar/mar_base/checkpoint-last.pth"
    if not os.path.exists(download_path) or overwrite:
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        os.makedirs("pretrained_models/mar/mar_base", exist_ok=True)
        r = requests.get("https://www.dropbox.com/scl/fi/f6dpuyjb7fudzxcyhvrhk/checkpoint-last.pth?rlkey=a6i4bo71vhfo4anp33n9ukujb&dl=0", stream=True, headers=headers)
        print("Downloading MAR-B...")
        with open(download_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=1587):
                if chunk:
                    f.write(chunk)


def download_pretrained_marl(overwrite=False):
    download_path = "pretrained_models/mar/mar_large/checkpoint-last.pth"
    if not os.path.exists(download_path) or overwrite:
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        os.makedirs("pretrained_models/mar/mar_large", exist_ok=True)
        r = requests.get("https://www.dropbox.com/scl/fi/pxacc5b2mrt3ifw4cah6k/checkpoint-last.pth?rlkey=m48ovo6g7ivcbosrbdaz0ehqt&dl=0", stream=True, headers=headers)
        print("Downloading MAR-L...")
        with open(download_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=3650):
                if chunk:
                    f.write(chunk)


def download_pretrained_marh(overwrite=False):
    download_path = "pretrained_models/mar/mar_huge/checkpoint-last.pth"
    if not os.path.exists(download_path) or overwrite:
        headers = {'user-agent': 'Wget/1.16 (linux-gnu)'}
        os.makedirs("pretrained_models/mar/mar_huge", exist_ok=True)
        r = requests.get("https://www.dropbox.com/scl/fi/1qmfx6fpy3k7j9vcjjs3s/checkpoint-last.pth?rlkey=4lae281yzxb406atp32vzc83o&dl=0", stream=True, headers=headers)
        print("Downloading MAR-H...")
        with open(download_path, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=1024*1024), unit="MB", total=7191):
                if chunk:
                    f.write(chunk)


if __name__ == "__main__":
    download_pretrained_vae()
    download_pretrained_marb()
    download_pretrained_marl()
    download_pretrained_marh()
