import argparse
import json
import os
import time
import urllib.request
import urllib.error


def post_discord(webhook_url: str, content: str):
    data = json.dumps({"content": content}).encode("utf-8")
    req = urllib.request.Request(
        webhook_url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "AnyBimanual-R2BC-Progress-Watcher/0.1",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            return res.status
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"[discord] HTTPError: {e.code} {e.reason}")
        print(f"[discord] response body: {body}")
    except Exception as e:
        print(f"[discord] failed: {repr(e)}")

    return None

def format_record(record: dict) -> str:
    update = record.get("update")
    num_updates = record.get("num_updates")
    loss = record.get("loss")
    loss_delta = record.get("loss_delta")
    trans_loss = record.get("trans_loss")
    rot_loss = record.get("rot_loss")
    grip_loss = record.get("grip_loss")
    gt = record.get("last_gt_voxel")
    pred = record.get("last_pred_voxel")
    step_index = record.get("last_step_index")
    target_arm = record.get("target_arm")
    grad_accum_steps = record.get("grad_accum_steps")
    effective_batch_size = record.get("effective_batch_size")

    def f(x):
        if isinstance(x, float):
            return f"{x:.3f}"
        return str(x)

    return (
        f"📈 R2BC progress {update}/{num_updates}\n"
        f"arm={target_arm}, eff_batch={effective_batch_size}, accum={grad_accum_steps}\n"
        f"loss={f(loss)} delta={f(loss_delta)}\n"
        f"trans={f(trans_loss)} rot={f(rot_loss)} grip={f(grip_loss)}\n"
        f"last step={step_index} gt={gt} pred={pred}"
    )


def follow_jsonl(path: str):
    # ファイルができるまで待つ
    while not os.path.exists(path):
        print(f"[watch] waiting for {path}")
        time.sleep(2)

    with open(path, "r", encoding="utf-8") as f:
        # 既存分を最後まで読む。既存分も送りたいなら seek を消す
        f.seek(0, os.SEEK_END)

        while True:
            line = f.readline()
            if not line:
                time.sleep(1)
                continue

            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[watch] json decode failed: {e}: {line[:120]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--webhook-url", default=os.environ.get("DISCORD_WEBHOOK_URL"))
    parser.add_argument("--send-start", action="store_true")
    args = parser.parse_args()

    if not args.webhook_url:
        raise RuntimeError(
            "Discord webhook URL is missing. Set DISCORD_WEBHOOK_URL or pass --webhook-url."
        )

    print("[watch] path:", args.path)

    if args.send_start:
        post_discord(args.webhook_url, f"🚀 progress watcher started\n`{args.path}`")

    for record in follow_jsonl(args.path):
        content = format_record(record)
        print(content)
        post_discord(args.webhook_url, content)

        # 最後まで来たら終了通知
        if record.get("update") == record.get("num_updates"):
            post_discord(args.webhook_url, "✅ 学習終了！")
            print("[watch] done.")
            break


if __name__ == "__main__":
    main()
