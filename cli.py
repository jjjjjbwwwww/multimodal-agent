
# cli.py
from __future__ import annotations

import argparse
from pathlib import Path

from agent_sdk import AgentConfig, run_agent, save_trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="path to an image")
    p.add_argument("--goal", required=True, help="what you want the agent to do")
    p.add_argument("--out_dir", default="runs", help="where to save trace files")
    p.add_argument("--test6_base", default="http://127.0.0.1:8006")
    p.add_argument("--model_dir", default=r"E:\python\test6\hf_tmp_blip_vqa")
    p.add_argument("--offline", default="true", choices=["true", "false"])
    p.add_argument("--max_new_tokens", type=int, default=40)
    args = p.parse_args()

    cfg = AgentConfig(
        test6_base=args.test6_base,
        model_dir=args.model_dir,
        offline=(args.offline.lower() == "true"),
        max_new_tokens=args.max_new_tokens,
    )

    out = run_agent(Path(args.image), args.goal, history=[], cfg=cfg)
    paths = save_trace(out, args.out_dir)

    print("✅ Done")
    print("trace_id:", out["trace_id"])
    print("saved json:", paths["json"])
    print("saved md  :", paths["md"])


if __name__ == "__main__":
    main()
