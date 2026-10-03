#!/usr/bin/env python3
"""训练逐 token 安全分类头（用户卖点：逐 token 分类，09-07 定案 = difflib 对级对齐标签 + 纯分类器先验）。

数据：build_token_labels.py 产物 jsonl，每行 {id, cwe, lang, code, labels[], side, n_unsafe}
      labels[i] ∈ {0=SAFE, 1=UNSAFE}，与 tok(code, add_special_tokens=False) 的 ids 等长。
流程：
  1. 骨干（mixed1_2 merged）no_grad 前向，按行提取 code 的 last hidden（与标签同 tokenize 对齐），
     特征常驻 CPU RAM（token_labels_multi_py 实际 ~12M tokens ×2B ≈ 86GB 超 80GB 卡, 0907 OOM 后改: 提取即 .cpu(), 训练/评估逐批上卡）
  2. TokenSecurityHead = 单 Linear(H, 2)（纯分类器先验，不加 MLP/Qformer 复杂化）
  3. token 级加权 CE（UNSAFE 反频权重）；按行 shuffle 的 batch，padding + 有效 mask
  4. hold-out 20% 行级 split 验证：token 级 acc/prec/recall/F1/AUC
     + 代码级判据（RL 奖励要用的是段级聚合，先验证定位能力）：
       - vul 行: 命中率 = true-UNSAFE span 内出现 pred-UNSAFE 的比例（逐 token 定位质量）
       - sec 行: 干净率 = 0 个 pred-UNSAFE 的比例（误报率）
保存 best（按 val F1_unsafe）→ out_dir/token_head.pt；报告 JSON → out_dir/report.json

用法：python3 -m secbart.train_token_head --data data_preparation/token_labels_multi_py.jsonl \
      --model saved-models/SCTRL/Qwen2.5-Coder-7B/btok_7b_fullft_mixed1_2/merged_hf_model \
      --out_dir <…>/seq_head_v3_token --gpu 4
"""
import argparse, json, random
from collections import Counter
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenSecurityHead(nn.Module):
    """逐 token 分类头：Linear(H, 2)，无 pooling —— 每个位置独立判 SAFE/UNSAFE。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.cls = nn.Linear(hidden_size, 2)

    def forward(self, feats):            # (B, L, H) -> (B, L, 2)
        return self.cls(feats)


def build_btoks_or_code(rows, tok, max_len):
    """纯代码 causal 模式（与 build_token_labels 同 tokenize 保证对齐）。"""
    all_ids = []
    for r in rows:
        ids = tok(r["code"], add_special_tokens=False,
                  truncation=True, max_length=max_len)["input_ids"]
        ids = ids[: len(r["labels"])]   # 保险: 与标签等长截齐
        assert len(ids) == len(r["labels"]), f"对齐失败 {r['id']}"
        all_ids.append(ids)
    return all_ids


def extract_features(model, tok, rows, device, max_len, bs=8):
    """分块 padding 前向提 last hidden；返回 [feat(B_i,L,H)] 列表与对应 ids 长度。
    仅模型前向，全 no_grad。"""
    code_ids = build_btoks_or_code(rows, tok, max_len)
    feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(code_ids), bs):
            chunk = code_ids[i:i + bs]
            bl = max(len(x) for x in chunk)
            pad = torch.zeros(len(chunk), bl, dtype=torch.long)
            for ni, x in enumerate(chunk):
                pad[ni, :len(x)] = torch.tensor(x)
            pad = pad.to(device)
            out = model(input_ids=pad, output_hidden_states=True)
            h = out.hidden_states[-1]                      # (B, bl, H) bf16
            for ni, x in enumerate(chunk):
                feats.append(h[ni, :len(x)].detach().cpu())  # 0907: 落 CPU, 全量 ~86GB 驻 GPU 必 OOM
    return feats


def metrics(tp, fp, fn, tn):
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return acc, prec, rec, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="build_token_labels.py 产物 jsonl")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=32, help="行数/step")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--max_len", type=int, default=1024)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in open(args.data)]
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, trn_rows = rows[:n_val], rows[n_val:]
    cnt_t = Counter()
    for r in trn_rows:
        cnt_t.update(r["labels"])
    print(f"rows train={len(trn_rows)} val={len(val_rows)}  "
          f"token分布: SAFE={cnt_t[0]} UNSAFE={cnt_t[1]} "
          f"(unsafe占比 {cnt_t[1]/max(cnt_t[0]+cnt_t[1],1):.3%})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"[feat] 骨干前向提取 train {len(trn_rows)} + val {len(val_rows)} 行 hidden...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(device)
    trn_feats = extract_features(model, tok, trn_rows, device, args.max_len)
    val_feats = extract_features(model, tok, val_rows, device, args.max_len)
    del model
    torch.cuda.empty_cache()
    print(f"[feat] 完成: train {len(trn_feats)} / val {len(val_feats)}，特征常驻 CPU, 逐批上卡")

    H = trn_feats[0].shape[-1]
    head = TokenSecurityHead(H).to(torch.bfloat16).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)

    w = torch.tensor([cnt_t[0] and 1.0 / cnt_t[0], 1.0 / max(cnt_t[1], 1)],
                     device=device, dtype=torch.bfloat16)
    w = w / w.sum() * 2
    print("token class weights:", w.tolist())

    def eval_head():
        head.eval()
        tp = fp = fn = tn = 0          # token 级 (UNSAFE=pos)
        vul_hit = sec_clean = 0
        v_tot = s_tot = 0
        logits_all, lab_all = [], []
        with torch.no_grad():
            for (f, r) in zip(val_feats, val_rows):
                logits = head(f.unsqueeze(0).to(device))[0]   # (L,2)
                p = logits.argmax(-1)
                lab = torch.tensor(r["labels"], device=device)
                tp += ((p == 1) & (lab == 1)).sum().item()
                fp += ((p == 1) & (lab == 0)).sum().item()
                fn += ((p == 0) & (lab == 1)).sum().item()
                tn += ((p == 0) & (lab == 0)).sum().item()
                logits_all.append(logits.float().cpu())
                lab_all.append(r["labels"])
                if r["side"] == "vul":          # 代码级定位判据
                    v_tot += 1
                    true_pos = [i for i, x in enumerate(r["labels"]) if x == 1]
                    hit = any(p[i].item() == 1 for i in true_pos)
                    vul_hit += int(hit)
                else:                            # sec 侧: 干净率
                    s_tot += 1
                    sec_clean += int((p == 1).sum().item() == 0)
        acc, prec, rec, f1 = metrics(tp, fp, fn, tn)
        # token AUC (UNSAFE=pos)
        import numpy as np
        lp = torch.cat(logits_all, 0)
        prob = torch.softmax(lp, -1)[:, 1].numpy()
        labn = np.concatenate([np.array(x) for x in lab_all])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labn, prob) if len(set(labn.tolist())) > 1 else float("nan")
        return dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc,
                    hit=vul_hit / max(v_tot, 1), clean=sec_clean / max(s_tot, 1),
                    vul_hit=int(vul_hit), v_tot=v_tot,
                    sec_clean=int(sec_clean), s_tot=s_tot)

    best = -1.0
    best_m = None
    for ep in range(args.epochs):
        head.train()
        idx = list(range(len(trn_feats)))
        random.shuffle(idx)
        tot = n = 0.0
        for i in range(0, len(idx), args.batch_size):
            bidx = idx[i:i + args.batch_size]
            bl = max(trn_feats[j].shape[0] for j in bidx)
            feats = torch.zeros(len(bidx), bl, H, dtype=trn_feats[0].dtype)
            for ni, j in enumerate(bidx):
                L = trn_feats[j].shape[0]
                feats[ni, :L] = trn_feats[j]
            feats = feats.to(device)
            lab = torch.full((len(bidx), bl), -100, dtype=torch.long)
            for ni, j in enumerate(bidx):
                L = trn_feats[j].shape[0]
                lab[ni, :L] = torch.tensor(trn_rows[j]["labels"])
            lab = lab.to(device)
            logits = head(feats)                          # (B, L, 2)
            loss = F.cross_entropy(logits.reshape(-1, 2), lab.reshape(-1),
                                   weight=w, ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(bidx)
            n += len(bidx)
        m = eval_head()
        if m["f1"] > best:
            best = m["f1"]
            best_m = m
            torch.save(head.state_dict(), out / "token_head.pt")
        if ep % 5 == 0 or ep == args.epochs - 1:
            print(f"ep {ep}: loss={tot/n:.4f} | token acc={m['acc']:.4f} "
                  f"prec={m['prec']:.4f} rec={m['rec']:.4f} f1={m['f1']:.4f} "
                  f"auc={m['auc']:.4f} | 代码级: vul命中={m['hit']:.4f} "
                  f"sec干净={m['clean']:.4f}")

    print("\n=== 逐 token 头最终报告 ===")
    print(f"val 行: vul={best_m['v_tot']} sec={best_m['s_tot']}")
    print(f"token级: acc={best_m['acc']:.4f} prec={best_m['prec']:.4f} "
          f"rec={best_m['rec']:.4f} F1={best_m['f1']:.4f} AUC={best_m['auc']:.4f}")
    print(f"代码级定位: vul 行 span 命中率={best_m['hit']:.4f} "
          f"(出现≥1 pred-UNSAFE 且落在真 span 内)  sec 行干净率={best_m['clean']:.4f}")
    with open(out / "report.json", "w") as f:
        json.dump({k: (v if not isinstance(v, (float, int)) else round(v, 4))
                   for k, v in best_m.items()}, f, indent=1)
    print(f"head 保存: {out / 'token_head.pt'}  报告: {out / 'report.json'}")
    print("上线门槛（RL 前先验证据）: token AUC>=0.90 且 vul span 命中>=0.85 "
          "且 sec 干净率>=0.95；不达标→扩标签数据/换骨干层")


if __name__ == "__main__":
    main()
