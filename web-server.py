import sys
import json
import asyncio

from aiohttp import web
import logging

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("web.log"),
        logging.StreamHandler()
    ]
)


logging.info("started")

try:
    from sentence_parser import STYPE_SEC, STYPE_AUX, PRIM_GL, SEC_GL, AUX_GL
    from network import into_one_hot, generate_batch, load_from_save

    enc, sec_dec, aux_dec, *_ = load_from_save()

    LOADED = True
except FileNotFoundError as e:
    logging.warning(f"Network not loaded: {e}")

    LOADED = False

def make_json_response(data, status=200):
    return web.Response(
        body=json.dumps(data),
        content_type='application/json',
        status=status,
    )


class WebInterface:
    def __init__(self, app):
        self.app = app

        app.router.add_post("/api/translate", self.translate)

        app.router.add_get("/", self.index)

        app.router.add_static("/", 'static')

        self.currently_blocked_users = set()

    async def index(self, req):
        return web.Response(
            body=open("static/index.html", "r").read(),
            content_type='text/html',
            status=200,
        )

    async def translate(self, req):
        if not LOADED:
            return make_json_response({"error": "Network not loaded. Please contact coral if this happens."}, status=500)
        ip, port = req.transport.get_extra_info("peername")
        if ip in self.currently_blocked_users:
            logging.info(f"Too many requests for {ip}")
            return make_json_response({"error": "Too many requests. Try again in a few seconds!"}, status=400)

        self.currently_blocked_users.add(ip)
        try:
            await asyncio.sleep(1)

            data = await req.json()

            bpe = PRIM_GL.str_to_bpe(data["input"])
            xs = torch.LongTensor([bpe])

            confidence_boost = data.get("confidence_boost", 1)
            confidence_boost = min(3, max(-3, float(confidence_boost)))

            logging.info(f"Translating {repr(data)}, confidence boost = {confidence_boost}")

            eof_idx = -1
            did_cuttof = False
            for e in range(5):
                ylen = 5 * 2 ** e
                ys = torch.LongTensor([[-1] * ylen])

                hid = enc(xs)
                outs, atts, hard_outs = sec_dec(hid, ys, teacher_forcing_prob=0, choice=True, confidence_boost=confidence_boost)
                out, att, hard_out = outs[0], atts[0], hard_outs[0]

                hout_eofs = (hard_out == SEC_GL.n_tokens() - 1).nonzero()
                if len(hout_eofs) == 0:
                    eof_idx = ylen
                    continue
                eof_idx = hout_eofs[0]
                did_cuttof = True

            out = torch.exp(out[:eof_idx])
            out /= out.sum(axis=1).unsqueeze(1)
            hard_out = hard_out[:eof_idx]
            att = att[:eof_idx]

            confidences = torch.gather(out, 1, hard_out.view(-1, 1))
            confidence = confidences.prod().item()

            hy_words = [SEC_GL.bpe_to_str([word]) for word in hard_out]

            if did_cuttof:
                out = "".join(hy_words)
            else:
                out = "".join(hy_words) + "..."

            logging.info(f"Got {repr(out)}, conf = {confidence}")
            return make_json_response({"result": out, "confidence": confidence})
        finally:
            self.currently_blocked_users.remove(ip)

    def run(self, port):
        web.run_app(self.app, port=port)


app = web.Application()

WEB_STATE = WebInterface(app)

if len(sys.argv) == 2:
    WEB_STATE.run(port=int(sys.argv[1]))
else:
    WEB_STATE.run(port=8080)
