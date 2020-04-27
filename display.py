import torch
import numpy as np

import matplotlib.pyplot as plt

from sentence_parser import STYPE_SEC, STYPE_AUX, PRIM_GL, SEC_GL, AUX_GL
from network import into_one_hot, generate_batch, load_from_save

enc, sec_dec, aux_dec, _ = load_from_save()

sec_info = ("sec", SEC_GL, sec_dec, STYPE_SEC)
aux_info = ("aux", AUX_GL, aux_dec, STYPE_AUX)

for name, gl, dec, stype in [sec_info, aux_info]:
    xs, ys = generate_batch(6, stype)

    hid = enc(xs)
    outs, atts = dec(hid, ys, 0.5)

    for i in range(len(xs)):
        plt.subplot(2, 3, i + 1)
        x = xs[i]

        y = ys[i]
        out = outs[i]
        att = atts[i]

        hard_out = out.argmax(axis=1)

        x_eofs = (x == -1).nonzero()
        if len(x_eofs) > 0:
            x = x[:x_eofs[0]]
            att = att[:, :x_eofs[0]]

        hout_eofs = (hard_out == gl.n_tokens() - 1).nonzero()
        if len(hout_eofs) > 0:
            y = y[:hout_eofs[0]]
            out = out[:hout_eofs[0]]
            hard_out = hard_out[:hout_eofs[0]]
            att = att[:hout_eofs[0]]

        x_words = [PRIM_GL.bpe_to_str([word]) for word in x]
        y_words = [gl.bpe_to_str([word]) for word in hard_out]

        print("/".join(x_words), "/".join(y_words))
        plt.imshow(att.detach().numpy())
        plt.xticks(np.arange(len(x)), x_words, rotation="vertical")
        plt.yticks(np.arange(len(y_words)), y_words)

    plt.show()
