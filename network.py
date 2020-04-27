import os, sys
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_parser import load_one_pair, STYPE_SEC, PRIM_GL, SEC_GL

torch.set_printoptions(precision=5)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

print("starting using device", device)

ENCODER_SAVE = "save/encoder.pth"
DECODER_SAVE = "save/decoder.pth"
OPT_SAVE     = "save/opt.pth"

class Encoder(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size):
        super(Encoder, self).__init__()

        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.embedding = nn.Embedding(input_size, emb_size).to(device)

        self.rnn_right = nn.Linear(emb_size + hidden_size, hidden_size).to(device)
        self.rnn_left = nn.Linear(emb_size + hidden_size, hidden_size).to(device)

    def run(self, inp, run_right):
        current_hid = self.init_hidden(inp.size(0))

        hidden_states = torch.FloatTensor(inp.size(0), inp.size(1), self.hidden_size).to(device)

        network_to_use = self.rnn_right if run_right else self.rnn_left

        for i in range(inp.size(1)):
            if not run_right:
                i = inp.size(1) - i - 1
            current_ch = inp[:, i]

            emb = self.embedding(current_ch % self.input_size)

            current_hid = network_to_use(torch.cat([emb, current_hid], axis=1))
            current_hid = F.elu(current_hid)
            hidden_states[:, i] = current_hid

        return hidden_states

    def forward(self, inp):
        hidden_right = self.run(inp, True)
        hidden_left = self.run(inp, False)

        return torch.cat([hidden_right, hidden_left], axis=2)

    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size)).to(device)

class Decoder(nn.Module):
    def __init__(self, output_size, emb_size, enc_hid_size, dec_hid_size):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.enc_hid_size = enc_hid_size
        self.dec_hid_size = dec_hid_size
        self.output_size = output_size

        self.energizer_l1 = nn.Linear(enc_hid_size + dec_hid_size, 20)
        self.energizer_l2 = nn.Linear(20, 1)

        self.embedding = nn.Embedding(output_size, emb_size)

        self.rnn = nn.Linear(emb_size + dec_hid_size + enc_hid_size, dec_hid_size)
        self.out = nn.Linear(emb_size + dec_hid_size + enc_hid_size, output_size)

    def forward(self, enc_hid, real_output, teacher_forcing_prob=0.5):
        batch_size = enc_hid.size(0)
        inp_size = enc_hid.size(1)
        out_size = real_output.size(1)

        last_char = torch.LongTensor(batch_size).to(device).zero_()
        last_hidden = self.init_hidden(batch_size)

        output = torch.FloatTensor(batch_size, out_size, self.output_size).to(device)
        weights_mat = torch.FloatTensor(batch_size, out_size, inp_size).to(device)

        for i in range(out_size):
            # Generate energies

            energies = torch.FloatTensor(batch_size, inp_size).to(device)

            for j in range(inp_size):
                j_energies = self.energizer_l1(torch.cat([enc_hid[:, j], last_hidden], axis=1))
                j_energies = F.elu(j_energies)
                j_energies = self.energizer_l2(j_energies)
                j_energies = F.elu(j_energies)
                energies[:, j] = j_energies[:, 0]

            summed = torch.exp(energies).sum(axis=1)

            weights = torch.exp(energies) / torch.unsqueeze(summed, 1).repeat(1, inp_size)

            weights_mat[:, i] = weights

            weights = torch.unsqueeze(weights, 2).repeat(1, 1, enc_hid.size(2))
            context = (enc_hid * weights).sum(axis=1)

            last_ch = self.embedding(last_char % self.output_size)

            new_hidden = self.rnn(torch.cat([last_ch, last_hidden, context], axis=1))
            new_hidden = F.elu(new_hidden)

            out = self.out(torch.cat([last_ch, new_hidden, context], axis=1))
            out = F.elu(out)
            output[:, i] = out
            last_hidden = new_hidden

            # Update last_char
            if random.random() < teacher_forcing_prob:
                # Teacher forcing
                last_char_idx = real_output[:, i]
            else:
                last_char_idx = out.argmax(axis=1)

            last_char = last_char_idx.clone()

        return output, weights_mat


    def init_hidden(self, batch_size):
        return torch.zeros((batch_size, self.dec_hid_size)).to(device)

def into_one_hot(values, n_tokens):
    one_hot = torch.LongTensor(values.shape[0], values.shape[1], n_tokens, device=device).zero_()
    for i in range(values.size(0)):
        one_hot[i, torch.arange(values.size(1)), values[i] % n_tokens] = 1
    return one_hot

def generate_batch(batch_size, other_stype, max_length=None):
    xs, ys = [], []

    if max_length is None:
        max_length = 3 + random.expovariate(1 / 10)
        max_length = min(15, max(3, max_length))

    min_length = int(max_length * 0.9 - 2)

    longest_x = longest_y = 0
    for i in range(batch_size):
        y = None
        while y is None or not (min_length < len(y) < max_length or min_length < len(x) < max_length):
            x, y = load_one_pair(other_stype)

        longest_x = max(longest_x, len(x))
        longest_y = max(longest_y, len(y))

        xs.append(x)
        ys.append(y)

    # Pad each sentence to the appropriate length
    for x in xs:
        x += [-1] * (longest_x - len(x))

    for y in ys:
        y += [-1] * (longest_y - len(y))

    x_tensors = []
    y_tensors = []

    for x in xs:
        x_tensor = torch.LongTensor(x)
        x_tensors.append(x_tensor.view(1, -1))

    for y in ys:
        y_tensor = torch.LongTensor(y)
        y_tensors.append(y_tensor.view(1, -1))

    return torch.cat(x_tensors, dim=0).to(device), torch.cat(y_tensors, dim=0).to(device)

ENCODER_SAVE = "save/enc.pth"
DECODER_SAVE = "save/dec.pth"

if __name__ == "__main__":
    enc = Encoder(PRIM_GL.n_tokens(), 20, 20).to(device)
    sec_dec = Decoder(SEC_GL.n_tokens(), 20, 40, 80).to(device)

    prim_sent, sec_sent = generate_batch(256, STYPE_SEC) #load_one_pair(STYPE_SEC)

    print("Prim :", prim_sent.size())
    print("Sec :", sec_sent.size())

    hid = enc(prim_sent)
    print("Hidden:", hid.size())

    dec, weights = sec_dec(hid, sec_sent)
    print("Dec :", dec.size())
