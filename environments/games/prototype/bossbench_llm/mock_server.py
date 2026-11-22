"""A stand-in OpenAI-compatible endpoint for checking the harness without a model.

  python -m bossbench_llm.mock_server --port 8009
  python -m bossbench_llm.run --case terraria/king_slime --tier easy --policy chat \
      --base-url http://127.0.0.1:8009/v1 --model mock --max-steps 20

It checks each request the way a real server would see it (system prompt first, one image per user turn,
alternating turns) and answers with a fixed-pattern action in the format the system prompt asks for. It is not a
policy: it does not look at the image.
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import sys

LUMINE = ["<|action_start|>0 0 0 ; D ; D ; D ; D ; D ; D<|action_end|>",
          "<|action_start|>40 0 0 ; A Space ; A Space ; A ; A ; LB ; LB<|action_end|>",
          "<|action_start|>-40 0 0 ; J ; J ; ; ; Space ; Space<|action_end|>"]


def check(messages):
    if not messages or messages[0].get("role") != "system":
        return "first message must be the system prompt"
    roles = [m.get("role") for m in messages[1:]]
    if not roles or roles[-1] != "user":
        return "last message must be a user turn"
    for a, b in zip(roles, roles[1:]):
        if a == b:
            return "user/assistant turns must alternate"
    for m in messages[1:]:
        if m["role"] == "user":
            images = [p for p in m.get("content", []) if isinstance(p, dict) and p.get("type") == "image_url"]
            if len(images) != 1 or not images[0]["image_url"]["url"].startswith("data:image/jpeg;base64,"):
                return "each user turn must carry exactly one JPEG data URL"
    return None


class Handler(BaseHTTPRequestHandler):
    counter = 0

    def log_message(self, *args):
        pass

    def do_POST(self):
        if not self.path.endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        problem = check(body.get("messages", []))
        if problem:
            out = json.dumps({"error": {"message": problem}}).encode()
            self.send_response(400)
        else:
            Handler.counter += 1
            system = body["messages"][0]["content"]
            if "<|action_start|>" in system:
                text = LUMINE[Handler.counter % len(LUMINE)]
            else:
                text = "{}"
            out = json.dumps({"id": f"mock-{Handler.counter}", "object": "chat.completion", "model": body.get("model"),
                              "choices": [{"index": 0, "finish_reason": "stop",
                                           "message": {"role": "assistant", "content": text}}],
                              "usage": {"prompt_tokens": 0, "completion_tokens": 0,
                                        "images": sum(1 for m in body["messages"] if m["role"] == "user")}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m bossbench_llm.mock_server")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8009)
    args = p.parse_args(argv)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock endpoint on http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
