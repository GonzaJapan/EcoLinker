import argparse
import random
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


def row_norm(matrix: torch.Tensor) -> torch.Tensor:
    row_sum = matrix.sum(dim=1, keepdim=True).clamp_min(1.0)
    return matrix / row_sum


def load_excel_tables(data_dir: str):
    if not data_dir or not data_dir.strip():
        raise ValueError("データディレクトリが未指定です。--data-dir を指定してください。")

    data_path = Path(data_dir)
    process_input_output = pd.read_excel(data_path / "プロセス入出力_20250730.xlsx")
    process = pd.read_excel(data_path / "プロセス_IDEA35確認済_20250709.xlsx")
    product = pd.read_excel(data_path / "製品_20250717.xlsx")
    return process_input_output, process, product


def build_process_text(row: pd.Series) -> str:
    title = str(row.get("プロセス名（日本語）", "")).strip()
    scope = str(row.get("技術の範囲（日本語）", "")).strip()
    return f"プロセス名: {title}\n技術の範囲: {scope}"


def build_product_text(row: pd.Series) -> str:
    name = str(row.get("製品名（日本語）", "")).strip()
    return f"製品: {name}"


def build_graph_from_tables(
    process_input_output: pd.DataFrame,
    process: pd.DataFrame,
    product: pd.DataFrame,
    device: torch.device,
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
):
    from sentence_transformers import SentenceTransformer

    process_uuid_col = "*プロセスUUID"
    product_uuid_col = "*製品UUID"
    pio_process_col = "*プロセスUUID_LEC"
    pio_product_col = "製品/基本フローUUID_LEC"
    pio_dir_col = "*方向"

    process_uuid_list = process[process_uuid_col].tolist()
    product_uuid_list = product[product_uuid_col].tolist()
    proc2idx = {u: i for i, u in enumerate(process_uuid_list)}
    prod2idx = {u: i for i, u in enumerate(product_uuid_list)}

    pio = process_input_output[[pio_process_col, pio_product_col, pio_dir_col]].dropna()
    pio = pio[
        pio[pio_process_col].isin(proc2idx.keys())
        & pio[pio_product_col].isin(prod2idx.keys())
    ]

    process_consumes = pio[pio[pio_dir_col] == "入力"]
    process_produces = pio[pio[pio_dir_col] == "出力"]

    consumes_raw = list(zip(process_consumes[pio_product_col], process_consumes[pio_process_col]))
    produces_raw = list(zip(process_produces[pio_process_col], process_produces[pio_product_col]))

    consumes_edges: List[Tuple[int, int]] = sorted(
        set((prod2idx[prod_uuid], proc2idx[proc_uuid]) for prod_uuid, proc_uuid in consumes_raw)
    )
    produces_edges: List[Tuple[int, int]] = sorted(
        set((proc2idx[proc_uuid], prod2idx[prod_uuid]) for proc_uuid, prod_uuid in produces_raw)
    )

    embedding_model = SentenceTransformer(embedding_model_name)

    process_texts = [build_process_text(row) for _, row in process.iterrows()]
    product_texts = [build_product_text(row) for _, row in product.iterrows()]

    process_emb = embedding_model.encode(
        process_texts,
        batch_size=64,
        normalize_embeddings=True,
    )
    product_emb = embedding_model.encode(
        product_texts,
        batch_size=64,
        normalize_embeddings=True,
    )

    hp0 = torch.tensor(process_emb, dtype=torch.float32, device=device)
    hq0 = torch.tensor(product_emb, dtype=torch.float32, device=device)

    np_nodes = hp0.size(0)
    nq_nodes = hq0.size(0)

    a_cons_pre = torch.zeros((np_nodes, nq_nodes), device=device)
    for q, p in consumes_edges:
        a_cons_pre[p, q] += 1.0
    a_cons = row_norm(a_cons_pre)

    a_prod_pre = torch.zeros((nq_nodes, np_nodes), device=device)
    for p, q in produces_edges:
        a_prod_pre[q, p] += 1.0
    a_prod = row_norm(a_prod_pre)

    a_rev_cons = row_norm(a_cons_pre.T)
    a_rev_prod = row_norm(a_prod_pre.T)

    return hp0, hq0, consumes_edges, produces_edges, a_cons, a_prod, a_rev_cons, a_rev_prod


class TwoLayerHeteroSAGE(nn.Module):
    def __init__(self, dp_in: int, dq_in: int, d1: int, d2: int, dropout: float,
                 a_cons: torch.Tensor, a_prod: torch.Tensor,
                 a_rev_cons: torch.Tensor, a_rev_prod: torch.Tensor):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.register_buffer("a_cons", a_cons)
        self.register_buffer("a_prod", a_prod)
        self.register_buffer("a_rev_cons", a_rev_cons)
        self.register_buffer("a_rev_prod", a_rev_prod)

        self.wp1 = nn.Linear(dp_in, d1, bias=True)
        self.wq1 = nn.Linear(dq_in, d1, bias=True)
        self.wcons1 = nn.Linear(dq_in, d1, bias=True)
        self.wrprod1 = nn.Linear(dq_in, d1, bias=True)
        self.wprod1 = nn.Linear(dp_in, d1, bias=True)
        self.wrcons1 = nn.Linear(dp_in, d1, bias=True)

        self.wp2 = nn.Linear(d1, d2, bias=True)
        self.wq2 = nn.Linear(d1, d2, bias=True)
        self.wcons2 = nn.Linear(d1, d2, bias=True)
        self.wrprod2 = nn.Linear(d1, d2, bias=True)
        self.wprod2 = nn.Linear(d1, d2, bias=True)
        self.wrcons2 = nn.Linear(d1, d2, bias=True)

        self.r_cons = nn.Linear(d2, d2, bias=False)
        self.r_prod = nn.Linear(d2, d2, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, hp: torch.Tensor, hq: torch.Tensor):
        hp1 = self.wp1(hp) + self.a_cons @ self.wcons1(hq) + self.a_rev_prod @ self.wrprod1(hq)
        hp1 = self.dropout(F.leaky_relu(hp1, negative_slope=0.1))

        hq1 = self.wq1(hq) + self.a_prod @ self.wprod1(hp) + self.a_rev_cons @ self.wrcons1(hp)
        hq1 = self.dropout(F.leaky_relu(hq1, negative_slope=0.1))

        hp2 = self.wp2(hp1) + self.a_cons @ self.wcons2(hq1) + self.a_rev_prod @ self.wrprod2(hq1)
        hp2 = F.leaky_relu(hp2, negative_slope=0.1)

        hq2 = self.wq2(hq1) + self.a_prod @ self.wprod2(hp1) + self.a_rev_cons @ self.wrcons2(hp1)
        hq2 = F.leaky_relu(hq2, negative_slope=0.1)

        hp2 = hp2 / (hp2.norm(dim=1, keepdim=True) + 1e-9)
        hq2 = hq2 / (hq2.norm(dim=1, keepdim=True) + 1e-9)
        return hp2, hq2

    def score_consumes(self, hq: torch.Tensor, hp: torch.Tensor, pairs):
        q = torch.tensor([qp[0] for qp in pairs], device=hp.device)
        p = torch.tensor([qp[1] for qp in pairs], device=hp.device)
        zq = hq[q] @ self.r_cons.weight.T
        zp = hp[p]
        return torch.sigmoid((zq * zp).sum(dim=-1))

    def score_produces(self, hp: torch.Tensor, hq: torch.Tensor, pairs):
        p = torch.tensor([pq[0] for pq in pairs], device=hp.device)
        q = torch.tensor([pq[1] for pq in pairs], device=hp.device)
        zp = hp[p] @ self.r_prod.weight.T
        zq = hq[q]
        return torch.sigmoid((zp * zq).sum(dim=-1))


def sample_neg(existing_pairs, left_size: int, right_size: int, is_q_to_p: bool):
    existing = set(existing_pairs)
    negatives = []
    target = max(1, len(existing_pairs))
    tries = 0

    while len(negatives) < target and tries < 10000:
        if is_q_to_p:
            q = random.randrange(right_size)
            p = random.randrange(left_size)
            candidate = (q, p)
        else:
            p = random.randrange(left_size)
            q = random.randrange(right_size)
            candidate = (p, q)

        if candidate not in existing:
            negatives.append(candidate)
        tries += 1

    return negatives


def train(model: TwoLayerHeteroSAGE, hp0: torch.Tensor, hq0: torch.Tensor,
          consumes_edges, produces_edges,
          epochs: int = 600, lr: float = 0.02, wd: float = 1e-4):
    np_nodes = hp0.size(0)
    nq_nodes = hq0.size(0)

    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        hp2, hq2 = model(hp0, hq0)

        neg_c = sample_neg(consumes_edges, np_nodes, nq_nodes, is_q_to_p=True)
        s_pos_c = model.score_consumes(hq2, hp2, consumes_edges)
        s_neg_c = model.score_consumes(hq2, hp2, neg_c)
        y_c = torch.cat([torch.ones_like(s_pos_c), torch.zeros_like(s_neg_c)])
        s_c = torch.cat([s_pos_c, s_neg_c])
        loss_c = bce(s_c, y_c)

        neg_p = sample_neg(produces_edges, np_nodes, nq_nodes, is_q_to_p=False)
        s_pos_p = model.score_produces(hp2, hq2, produces_edges)
        s_neg_p = model.score_produces(hp2, hq2, neg_p)
        y_p = torch.cat([torch.ones_like(s_pos_p), torch.zeros_like(s_neg_p)])
        s_p = torch.cat([s_pos_p, s_neg_p])
        loss_p = bce(s_p, y_p)

        loss = loss_c + loss_p
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch {epoch:3d}  loss={loss.item():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN training with private Excel source data")
    parser.add_argument(
        "--data-dir",
        default="",
        help="Excelファイルがあるディレクトリ（デフォルト空文字）",
    )
    parser.add_argument(
        "--embedding-model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name",
    )
    args = parser.parse_args()

    torch.manual_seed(123)
    random.seed(123)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 実データの読み込み
    data_dir = args.data_dir
    process_input_output, process, product = load_excel_tables(data_dir)
    print("Loaded Excel tables:", process_input_output.shape, process.shape, product.shape)

    hp0, hq0, consumes_edges, produces_edges, a_cons, a_prod, a_rev_cons, a_rev_prod = build_graph_from_tables(
        process_input_output=process_input_output,
        process=process,
        product=product,
        device=device,
        embedding_model_name=args.embedding_model_name,
    )
    print(f"hp0 shape: {hp0.shape}, hq0 shape: {hq0.shape}")
    print(f"consumes_edges: {len(consumes_edges)}, produces_edges: {len(produces_edges)}")

    model = TwoLayerHeteroSAGE(
        dp_in=hp0.size(1),
        dq_in=hq0.size(1),
        d1=16,
        d2=16,
        dropout=0.1,
        a_cons=a_cons,
        a_prod=a_prod,
        a_rev_cons=a_rev_cons,
        a_rev_prod=a_rev_prod,
    ).to(device)

    train(model, hp0, hq0, consumes_edges, produces_edges)

    model.eval()
    with torch.no_grad():
        hp2, hq2 = model(hp0, hq0)
        s_prod = torch.sigmoid((hp2 @ model.r_prod.weight.T) @ hq2.T)
        s_cons = torch.sigmoid((hq2 @ model.r_cons.weight.T) @ hp2.T)

    print("\nProduces scores (rows=P, cols=Q):\n", s_prod.detach().cpu().numpy())
    print("\nConsumes  scores (rows=Q, cols=P):\n", s_cons.detach().cpu().numpy())
