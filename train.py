import random
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentence_parser import load_one_pair, STYPE_SEC, PRIM_GL, SEC_GL
from network import device, Encoder, Decoder, into_one_hot, ENCODER_SAVE, DECODER_SAVE, OPT_SAVE, generate_batch

def display_tokens(toklist, gl):
    out = ""
    last = 0 # 0 = any char, 1 = end, 2 = end after another end
    for tidx in toklist:
        tok = gl.bpe_to_str([tidx])
        if tok != "<EOF>":
            out += "/" + tok
            last = 0
        else:
            if last == 0:
                out += "/<EOF>"
                last += 1
            elif last == 1:
                out += "..."
                last += 1
    return out[1:]

enc = Encoder(PRIM_GL.n_tokens(), 20, 20)
dec = Decoder(SEC_GL.n_tokens(), 20, 40, 80)


if os.path.isfile(ENCODER_SAVE) and os.path.isfile(DECODER_SAVE) and os.path.isfile(OPT_SAVE):
    print("Loading from save")
    enc.load_state_dict(torch.load(ENCODER_SAVE, map_location=device))
    dec.load_state_dict(torch.load(DECODER_SAVE, map_location=device))

    enc = enc.to(device)
    dec = dec.to(device)

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=0.001,
    )

    opt.load_state_dict(torch.load(OPT_SAVE, map_location=device))

else:
    enc = enc.to(device)
    dec = dec.to(device)

    opt = torch.optim.Adam(
        list(enc.parameters()) + list(dec.parameters()),
        lr=0.001,
    )

EPSILON = 1e-8

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    crit = nn.CrossEntropyLoss()

    epoch = 0
    while True:
        epoch += 1

        losses = []
        for batch_nr in range(16):
            print(hex(batch_nr)[2:], end=" ")
            xs, ys = generate_batch(2048, STYPE_SEC)

            print("l={:2d}/{:2d};".format(xs.size(1), ys.size(1)), end=" ", flush=True)

            enc.zero_grad()
            dec.zero_grad()

            print("z", end="", flush=True)

            hids = enc(xs)
            print("e", end="", flush=True)
            y_hat, _ = dec(hids, ys)
            print("d", end="", flush=True)

            pred = y_hat.argmax(axis=2)
            acc = (pred == ys).type(torch.FloatTensor).mean()

            print("a", end=" ", flush=True)

            loss = crit(EPSILON + y_hat.view(-1, SEC_GL.n_tokens()), ys.view(-1) % SEC_GL.n_tokens())

            print("L={:.3f}; a={:.3f}%".format(loss, acc*100), end=" ")
            loss.backward()
            print("b", end=" ")
            opt.step()
            print("s")

            losses.append(loss.item())

        torch.save(enc.state_dict(), ENCODER_SAVE)
        torch.save(dec.state_dict(), DECODER_SAVE)
        torch.save(opt.state_dict(), OPT_SAVE)

        print("Epoch", epoch, "done")

        xs, ys = generate_batch(16, STYPE_SEC, max_length=10)

        hids = enc(xs)
        y_hat, _ = dec(hids, ys, teacher_forcing_prob=0)

        # Display
        for i in range(16):
            print()
            print(">", display_tokens(xs[i], PRIM_GL))
            print("=", display_tokens(ys[i], SEC_GL))
            gen = y_hat[i].argmax(dim=1)
            print("≈", display_tokens(gen, SEC_GL))

        print("Loss:", sum(losses) / len(losses))

