import os
from tqdm import tqdm
import requests
import time
import tarfile
import tempfile

SENTENCES_URL = "https://downloads.tatoeba.org/exports/sentences.tar.bz2"
LINKS_URL = "https://downloads.tatoeba.org/exports/links.tar.bz2"

CACHE_DIR = "~/.cache/ilo-pi-ante-toki/raw/"
TMP_DIR = tempfile.gettempdir()

CACHE_DIR = os.path.expanduser(CACHE_DIR)


if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def download_file(data_url, output_file):
    with requests.get(data_url, stream=True) as resp:

        size = int(resp.headers["Content-Length"])
        print(f"Loading {size} bytes from {data_url}")

        block_size = 1024 # 1 kiB

        with tqdm(total=size, unit="iB", unit_scale=True) as prog:

            start = time.time()
            for block in resp.iter_content(block_size):
                prog.update(len(block))
                output_file.write(block)

            took = time.time() - start

        print(f"Loaded {size} compressed bytes in {took:.4} seconds")

def extract_file(file_tar, output_file, file_to_extract):
    extracting = tarfile.open(fileobj=file_tar, mode="r:bz2")

    size = extracting.getmember(file_to_extract).size
    print(f"Extracting {size} bytes")

    buf = extracting.extractfile(file_to_extract)

    block_size = 1024

    with tqdm(total=size, unit="iB", unit_scale=True) as prog:
        start = time.time()
        while True:
            read = buf.read(block_size)
            prog.update(len(read))

            if len(read) == 0:
                break
            output_file.write(read)

        took = time.time() - start

    print(f"Extracted {size} bytes in {took:.4} seconds")



def plural(n, sing, plur):
    if n == 1:
        return f"{n} {sing}"
    return f"{n} {plur}"

def format_time(n_seconds):
    if n_seconds >= 60 * 60 * 24:
        # Days + hours
        major_length = 60 * 60 * 24
        minor_length = 60 * 60
        maj_sing, maj_plur = "day", "days"
        min_sing, min_plur = "hour", "hours"

    elif n_seconds >= 60 * 60:
        # Hours + minutes
        major_length = 60 * 60
        minor_length = 60
        maj_sing, maj_plur = "hour", "hours"
        min_sing, min_plur = "minute", "minutes"

    elif n_seconds >= 60:
        # Minutes + seconds
        major_length = 60
        minor_length = 1
        maj_sing, maj_plur = "minute", "minutes"
        min_sing, min_plur = "second", "seconds"

    else:
        # Just seconds
        return plural(int(n_seconds + 0.5), "second", "seconds")


    n_maj = int(n_seconds // major_length)
    n_min = (n_seconds / minor_length) % (major_length / minor_length)

    n_min = int(n_min + 0.5)

    s_maj = plural(n_maj, maj_sing, maj_plur)
    if n_min == 0:
        return s_maj

    s_min = plural(n_min, min_sing, min_plur)
    return f"{s_maj} and {s_min}"

compressed_sentence_path = os.path.join(TMP_DIR, "sentences.tar.bz2")
sentence_path = os.path.join(CACHE_DIR, "sentences.tsv")

compressed_link_path = os.path.join(TMP_DIR, "links.tar.bz2")
link_path = os.path.join(CACHE_DIR, "links.tsv")

if __name__ == "__main__":
    if os.path.isfile(sentence_path) and os.path.isfile(link_path):
        file_age = time.time() - os.path.getmtime(sentence_path)
        print(f"The compressed data is already downloaded, but {format_time(file_age)} old")
        if input("Redownload? [Y/n] ").lower() == "n":
            print("Quitting")
            exit()

    print("Loading sentences")
    with open(compressed_sentence_path, "bw") as comp_file:
        download_file(SENTENCES_URL, comp_file)

    with open(compressed_sentence_path, "rb") as comp_file, open(sentence_path, "wb") as out_file:
        extract_file(comp_file, out_file, "sentences.csv")

    print("Loading links")
    with open(compressed_link_path, "bw") as comp_file:
        download_file(LINKS_URL, comp_file)

    with open(compressed_link_path, "rb") as comp_file, open(link_path, "wb") as out_file:
        extract_file(comp_file, out_file, "links.csv")


    print("Done")
