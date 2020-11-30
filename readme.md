# ilo pi ante toki

ilo pi ante toki is a translator based on transfer learning on top of the method proposed by in Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et. al., arXiv:1409.0473 [cs.CL]).

The transfer is based on training the network to translate from English to Spanish (but this can be changed in the data-loading script), and building on that to translate from English to toki pona. This works well because the dataset of English -> Spanish is quite large (about 200000 pairs), which gives the network a lot of context about English. The dataset of English -> toki pona is a lot smaller, about 13000 pairs.

All sentences are loaded from [Tateoba](https://tatoeba.org), and are under the CC BY 2.0 FR license.

## Loading the data

The first step is to download the raw data from Tatoeba, which is done with a Python script. We load the `sentences.tar.bz2` and `links.tar.bz2` files, and untar them.

```sh
python3 load-data/data_loader.py
```

All files are by default put into the folder `cache/ilo-pi-ante-toki/`, which will be created automatically by `data_loader.py`. If you run Windows, you might need to change this path to something else. This needs to be done in all files separately.

The uncompressed data is quite large, around 450MiB for the sentences and 250MiB for the links. These include a lot of languages we don't need, and is stored in quite an inefficient format for reading arbitrary sentence pairs. The program `load-data/select-langs.rs` processes and converts this data into a more friendly format, only including the languages we want. In this script you can specify the languages to use. The primary language is the input to the translator, the secondary is the output and the auxiliary language is the transfor-learning part.

```sh
rustc -O load-data/select-langs.rs
./select-langs
```

This will run for a few minutes.

## Training the model

TODO
