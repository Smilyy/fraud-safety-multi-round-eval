import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TGNMemory


SUSPICIOUS_PATTERNS = {
    "payment": re.compile(r"\b(pay|payment|transfer|deposit|bank|wire|fee|funds?|crypto|wallet)\b", re.I),
    "urgency": re.compile(r"\b(urgent|immediately|today|deadline|expire|final notice|last chance|asap)\b", re.I),
    "credential": re.compile(r"\b(password|account|verify|verification|login|otp|code|identity|ssn)\b", re.I),
    "reward": re.compile(r"\b(job|salary|bonus|profit|investment|return|earn|commission)\b", re.I),
    "authority": re.compile(r"\b(police|court|government|official|agency|license|department|tax)\b", re.I),
    "emotion": re.compile(r"\b(friend|relationship|love|care|help me|trust|dear)\b", re.I),
}

REQUEST_TYPES = ["payment", "credential", "urgency", "reward", "authority", "emotion"]
NODE_TYPES = [
    "sender",
    "receiver",
    "organization",
    "category",
    "channel",
    "request",
    "round",
    "email",
    "phone",
    "url",
    "money",
    "identifier",
]
CATEGORY_PRIORS = {
    "phishing": 0.72,
    "impersonation": 0.70,
    "fraudulent service": 0.66,
    "fake job posting": 0.63,
    "network friendship": 0.61,
}
REQUEST_WEIGHTS = {
    "payment": 0.34,
    "credential": 0.34,
    "urgency": 0.12,
    "authority": 0.10,
    "reward": 0.06,
    "emotion": 0.04,
}
EDGE_GROUPS = ["structural", "temporal", "request", "entity"]
EMAIL_RE = re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>()]+", re.I)
PHONE_RE = re.compile(r"(?:(?:\+?\d[\d\s().-]{6,}\d))")
MONEY_RE = re.compile(
    r"(?:[$€£¥]\s?\d[\d,]*(?:\.\d+)?)|(?:\b\d[\d,]*(?:\.\d+)?\s?(?:usd|aud|eur|gbp|cny|yuan|dollars?)\b)",
    re.I,
)
IDENTIFIER_RE = re.compile(
    r"\b(?:[A-Z]{1,5}[-_])?[A-Z0-9]{3,}(?:[-_][A-Z0-9]{2,})+\b|\b\d{6,}\b",
    re.I,
)


def normalize_text_field(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(x) for x in value)
    if isinstance(value, dict):
        return " ".join(f"{k}:{v}" for k, v in value.items())
    return str(value)


def compact_organization_signal(value):
    text = normalize_text_field(value).strip()
    if " is " in text:
        text = text.split(" is ", 1)[0]
    text = text.split(",", 1)[0].strip()
    return text[:80]


def slugify(value):
    text = normalize_text_field(value).lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "unknown"


def keyword_hits(text: str):
    text = text or ""
    return {name: len(pattern.findall(text)) for name, pattern in SUSPICIOUS_PATTERNS.items()}


def request_types_from_text(text: str):
    hits = keyword_hits(text)
    return [k for k in REQUEST_TYPES if hits[k] > 0]


def signal_feature_vector(text: str):
    hits = keyword_hits(text)
    return np.array([hits[k] for k in REQUEST_TYPES], dtype=np.float32)


def base_event_features(example: dict, text: str):
    role_bg = example.get("role_bg", {})
    sender_company = normalize_text_field(role_bg.get("Sender_company", ""))
    sender_occ = normalize_text_field(role_bg.get("Sender_occupation", ""))
    text_hits = signal_feature_vector(text)
    meta = np.array(
        [
            CATEGORY_PRIORS.get(example.get("category", "").lower(), 0.60),
            1.0 if re.search(r"suspic|fraud|scam|unknown|fake", sender_company + " " + sender_occ, re.I) else 0.0,
            min(len(text) / 2000.0, 1.0),
            float(example.get("data_type", "") == "email"),
        ],
        dtype=np.float32,
    )
    return np.concatenate([text_hits, meta], dtype=np.float32)


def get_single_turn_message(example: dict):
    return normalize_text_field(example.get("generated text", example.get("generated_data", "")))


def get_multi_round_texts(example: dict):
    return [normalize_text_field(item.get("generated_data", "")) for item in example.get("multi-rounds fraud", [])]


def round_risk_score(text: str):
    hits = keyword_hits(text)
    severity = sum(min(hits[name], 2) * weight for name, weight in REQUEST_WEIGHTS.items())
    if hits["payment"] and hits["urgency"]:
        severity += 0.10
    if hits["credential"] and hits["authority"]:
        severity += 0.08
    if hits["payment"] and hits["authority"]:
        severity += 0.06
    return min(severity, 1.0), hits


def escalation_target(round_texts: list[str], round_index: int):
    score_series = [round_risk_score(text)[0] for text in round_texts]
    current_score = score_series[round_index]
    future_scores = score_series[round_index + 1 :]
    future_peak = max(future_scores) if future_scores else current_score
    escalation_gain = max(future_peak - current_score, 0.0)

    current_types = set(request_types_from_text(round_texts[round_index]))
    future_types = set()
    for future_text in round_texts[round_index + 1 :]:
        future_types.update(request_types_from_text(future_text))
    current_critical = float(bool({"payment", "credential"} & current_types))
    future_critical = float(bool({"payment", "credential"} & future_types))

    target = (
        0.45 * current_score
        + 0.25 * future_peak
        + 0.15 * escalation_gain
        + 0.10 * current_critical
        + 0.05 * future_critical
    )
    return round(min(target, 1.0), 4)


def normalize_entity_value(value: str):
    cleaned = normalize_text_field(value).strip().strip(".,;:()[]{}<>")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.lower()


def extract_entities(text: str):
    text = text or ""
    entities = {
        "email": sorted({normalize_entity_value(x) for x in EMAIL_RE.findall(text)}),
        "phone": sorted({normalize_entity_value(x) for x in PHONE_RE.findall(text)}),
        "url": sorted({normalize_entity_value(x) for x in URL_RE.findall(text)}),
        "money": sorted({normalize_entity_value(x) for x in MONEY_RE.findall(text)}),
        "identifier": sorted({normalize_entity_value(x) for x in IDENTIFIER_RE.findall(text)}),
    }
    # IDs often overlap with phone and amount strings; keep the id channel focused.
    filtered_ids = []
    overlaps = set(entities["phone"]) | set(entities["money"])
    for item in entities["identifier"]:
        if item not in overlaps and (len(item) >= 8 or "-" in item or "_" in item):
            filtered_ids.append(item)
    entities["identifier"] = filtered_ids[:8]
    entities["email"] = entities["email"][:6]
    entities["phone"] = entities["phone"][:6]
    entities["url"] = entities["url"][:6]
    entities["money"] = entities["money"][:8]
    return entities


def entity_feature_vector(entity_type: str, value: str, occurrence_count: int):
    feats = np.zeros(len(REQUEST_TYPES) + 4, dtype=np.float32)
    if entity_type == "money":
        feats[REQUEST_TYPES.index("payment")] = 1.0
    elif entity_type == "email":
        feats[REQUEST_TYPES.index("credential")] = 0.35
    elif entity_type == "url":
        feats[REQUEST_TYPES.index("credential")] = 0.25
    feats[len(REQUEST_TYPES) + 0] = min(len(value) / 64.0, 1.0)
    feats[len(REQUEST_TYPES) + 1] = min(occurrence_count / 4.0, 1.0)
    feats[len(REQUEST_TYPES) + 2] = 1.0 if any(token in value for token in ["bank", "pay", "login", "verify", "account"]) else 0.0
    feats[len(REQUEST_TYPES) + 3] = min(sum(char.isdigit() for char in value) / max(len(value), 1), 1.0)
    return feats


def edge_message_features(example: dict, text: str, edge_group: str, round_position: int, total_rounds: int):
    base = base_event_features(example, text)
    edge_group_vec = np.zeros(len(EDGE_GROUPS), dtype=np.float32)
    edge_group_vec[EDGE_GROUPS.index(edge_group)] = 1.0
    phase = np.array(
        [
            round_position / max(total_rounds, 1),
            1.0 if round_position == total_rounds else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([base, edge_group_vec, phase], dtype=np.float32)


def load_paired_examples(
    base_path: Path,
    levelup_path: Path,
    language: str,
    seed: int,
    test_fraction: float,
    split_manifest: Path | None = None,
):
    base_data = json.loads(base_path.read_text())
    levelup_data = json.loads(levelup_path.read_text())
    base_rows = [row for row in base_data if row.get("language", "").lower() == language.lower()]
    level_rows = [row for row in levelup_data if row.get("language", "").lower() == language.lower()]
    base_by_id = {row["id"]: row for row in base_rows}
    pair_by_id = {row["id"]: {"base": base_by_id[row["id"]], "levelup": row} for row in level_rows if row["id"] in base_by_id}

    if split_manifest and split_manifest.exists():
        manifest = json.loads(split_manifest.read_text())
        train_ids = manifest["train_ids"]
        test_ids = manifest["test_ids"]
    else:
        case_ids = list(pair_by_id)
        rng = random.Random(seed)
        rng.shuffle(case_ids)
        split = int(len(case_ids) * (1 - test_fraction))
        train_ids = case_ids[:split]
        test_ids = case_ids[split:]
        if split_manifest:
            split_manifest.parent.mkdir(parents=True, exist_ok=True)
            split_manifest.write_text(
                json.dumps(
                    {
                        "language": language,
                        "seed": seed,
                        "test_fraction": test_fraction,
                        "train_ids": train_ids,
                        "test_ids": test_ids,
                    },
                    indent=2,
                )
            )

    train_pairs = [pair_by_id[case_id] for case_id in train_ids if case_id in pair_by_id]
    test_pairs = [pair_by_id[case_id] for case_id in test_ids if case_id in pair_by_id]
    return train_pairs, test_pairs


def build_global_stats(train_pairs):
    org_counts, role_counts, sender_counts = {}, {}, {}
    for pair in train_pairs:
        ex = pair["base"]
        role_bg = ex.get("role_bg", {})
        org = slugify(role_bg.get("Sender_company", "unknown"))
        role = slugify(role_bg.get("Sender_occupation", "unknown"))
        sender = slugify(role_bg.get("Sender", "unknown"))
        org_counts[org] = org_counts.get(org, 0) + 1
        role_counts[role] = role_counts.get(role, 0) + 1
        sender_counts[sender] = sender_counts.get(sender, 0) + 1
    return {"org_counts": org_counts, "role_counts": role_counts, "sender_counts": sender_counts}


def node_type_one_hot(node_type: str):
    vec = np.zeros(len(NODE_TYPES), dtype=np.float32)
    vec[NODE_TYPES.index(node_type)] = 1.0
    return vec


@dataclass
class GraphSnapshot:
    x: torch.Tensor
    edge_index: torch.Tensor
    round_node_index: int
    round_node_indices: list[int]
    metadata: dict
    edge_events: list


def build_snapshot(example: dict, texts: list[str], global_stats: dict):
    role_bg = example.get("role_bg", {})
    org = slugify(role_bg.get("Sender_company", "unknown"))
    role = slugify(role_bg.get("Sender_occupation", "unknown"))
    sender = slugify(role_bg.get("Sender", "unknown"))
    receiver = slugify(role_bg.get("Receiver", "unknown"))
    category = slugify(example.get("category", "unknown"))
    channel = slugify(example.get("data_type", "message"))

    node_features = []
    edges = []
    edge_events = []
    metadata = {
        "sender": sender,
        "receiver": receiver,
        "organization": org,
        "role": role,
        "category": example.get("category", "unknown"),
        "channel": example.get("data_type", "message"),
        "entity_counts": {"email": 0, "phone": 0, "url": 0, "money": 0, "identifier": 0},
    }

    def add_node(ntype, feats):
        idx = len(node_features)
        node_features.append(np.concatenate([node_type_one_hot(ntype), feats], dtype=np.float32))
        return idx

    zero_sig = np.zeros(len(REQUEST_TYPES) + 4, dtype=np.float32)
    sender_idx = add_node("sender", zero_sig)
    receiver_idx = add_node("receiver", zero_sig)
    org_feat = np.array(
        [
            0, 0, 0, 0, 0, 0,
            min(global_stats["org_counts"].get(org, 1) / 20.0, 1.0),
            min(global_stats["role_counts"].get(role, 1) / 20.0, 1.0),
            min(global_stats["sender_counts"].get(sender, 1) / 20.0, 1.0),
            0.0,
        ],
        dtype=np.float32,
    )
    org_idx = add_node("organization", org_feat)
    cat_feat = np.array(
        [0, 0, 0, 0, 0, 0, CATEGORY_PRIORS.get(example.get("category", "").lower(), 0.60), 0, 0, 0],
        dtype=np.float32,
    )
    category_idx = add_node("category", cat_feat)
    channel_idx = add_node("channel", zero_sig)
    request_node_indices = {req: add_node("request", zero_sig) for req in REQUEST_TYPES}
    round_node_indices = []
    entity_node_indices = {name: {} for name in ["email", "phone", "url", "money", "identifier"]}
    entity_occurrence_counts = {name: {} for name in ["email", "phone", "url", "money", "identifier"]}

    def add_edge(src: int, dst: int, ridx: int, text: str, edge_group: str):
        edges.append((src, dst))
        edge_events.append(
            (
                src,
                dst,
                ridx + 1,
                edge_message_features(example, text, edge_group=edge_group, round_position=ridx + 1, total_rounds=len(texts)),
            )
        )

    def get_entity_node(entity_type: str, value: str):
        cache = entity_node_indices[entity_type]
        occurrence_cache = entity_occurrence_counts[entity_type]
        if value in cache:
            occurrence_cache[value] += 1
            node_idx = cache[value]
            node_features[node_idx] = np.concatenate(
                [node_type_one_hot(entity_type), entity_feature_vector(entity_type, value, occurrence_cache[value])],
                dtype=np.float32,
            )
            return node_idx
        occurrence_cache[value] = 1
        node_idx = add_node(entity_type, entity_feature_vector(entity_type, value, occurrence_cache[value]))
        cache[value] = node_idx
        metadata["entity_counts"][entity_type] += 1
        return node_idx

    for ridx, text in enumerate(texts):
        round_idx = add_node("round", base_event_features(example, text))
        round_node_indices.append(round_idx)
        base_edges = [
            (sender_idx, round_idx, "structural"),
            (round_idx, receiver_idx, "structural"),
            (sender_idx, org_idx, "structural"),
            (round_idx, category_idx, "structural"),
            (round_idx, channel_idx, "structural"),
        ]
        if ridx > 0:
            base_edges += [
                (round_node_indices[ridx - 1], round_idx, "temporal"),
                (round_idx, round_node_indices[ridx - 1], "temporal"),
            ]
        reqs = request_types_from_text(text)
        for req in reqs:
            req_idx = request_node_indices[req]
            base_edges += [(round_idx, req_idx, "request"), (req_idx, round_idx, "request")]

        entities = extract_entities(text)
        for entity_type, values in entities.items():
            for value in values:
                entity_idx = get_entity_node(entity_type, value)
                base_edges += [(round_idx, entity_idx, "entity"), (entity_idx, round_idx, "entity")]

        for src, dst, edge_group in base_edges:
            add_edge(src, dst, ridx, text, edge_group=edge_group)

    edge_index = torch.tensor(np.array(edges).T, dtype=torch.long) if edges else torch.empty((2, 0), dtype=torch.long)
    return GraphSnapshot(
        x=torch.tensor(np.stack(node_features), dtype=torch.float32),
        edge_index=edge_index,
        round_node_index=round_node_indices[-1],
        round_node_indices=round_node_indices,
        metadata=metadata,
        edge_events=edge_events,
    )


def iter_training_samples(train_pairs, global_stats):
    for pair in train_pairs:
        ex = pair["levelup"]
        round_texts = get_multi_round_texts(ex)
        texts = []
        for ridx, text in enumerate(round_texts):
            texts.append(text)
            snapshot = build_snapshot(ex, texts, global_stats)
            label = escalation_target(round_texts, ridx)
            yield snapshot, label


def peek_training_sample(train_pairs, global_stats):
    for sample in iter_training_samples(train_pairs, global_stats):
        return sample
    raise ValueError("no training samples available")


def create_static_model(train_pairs, global_stats, device: str = "cpu"):
    first_snapshot, _ = peek_training_sample(train_pairs, global_stats)
    return StaticGraphEncoder(first_snapshot.x.shape[1]).to(device)


def create_temporal_model(train_pairs, global_stats, device: str = "cpu", backbone: str = "gru"):
    first_snapshot, _ = peek_training_sample(train_pairs, global_stats)
    if backbone == "gru":
        return TemporalGraphEncoder(first_snapshot.x.shape[1]).to(device)
    if backbone == "tgn":
        max_nodes = 0
        for snapshot, _ in iter_training_samples(train_pairs, global_stats):
            max_nodes = max(max_nodes, snapshot.x.shape[0])
        raw_msg_dim = (
            len(first_snapshot.edge_events[0][3])
            if first_snapshot.edge_events
            else first_snapshot.x.shape[1] + len(EDGE_GROUPS) + 2
        )
        return TemporalTGNGraphEncoder(
            in_dim=first_snapshot.x.shape[1],
            raw_msg_dim=raw_msg_dim,
            max_nodes=max(max_nodes + 32, 256),
        ).to(device)
    raise ValueError(backbone)


class StaticGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, 1)

    def forward(self, snapshot: GraphSnapshot):
        x = snapshot.x
        edge_index = snapshot.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        prob = torch.sigmoid(self.cls(h[snapshot.round_node_index])).squeeze(-1)
        return prob, h


class TemporalGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward_snapshot(self, snapshot: GraphSnapshot, device: str):
        x = snapshot.x.to(device)
        edge_index = snapshot.edge_index.to(device)
        h = F.relu(self.conv1(x, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        seq = h[snapshot.round_node_indices].unsqueeze(0)
        out, _ = self.gru(seq)
        prob = torch.sigmoid(self.classifier(out[:, -1, :])).squeeze()
        return prob, out.squeeze(0)


class TemporalTGNGraphEncoder(nn.Module):
    def __init__(self, in_dim: int, raw_msg_dim: int, hidden_dim: int = 48, time_dim: int = 16, max_nodes: int = 256):
        super().__init__()
        self.max_nodes = max_nodes
        self.raw_msg_dim = raw_msg_dim
        self.memory_dim = hidden_dim
        self.memory = TGNMemory(
            num_nodes=max_nodes,
            raw_msg_dim=raw_msg_dim,
            memory_dim=hidden_dim,
            time_dim=time_dim,
            message_module=IdentityMessage(raw_msg_dim, hidden_dim, time_dim),
            aggregator_module=LastAggregator(),
        )
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward_snapshot(self, snapshot: GraphSnapshot, device: str):
        if snapshot.x.size(0) > self.max_nodes:
            raise ValueError(f"snapshot node count {snapshot.x.size(0)} exceeds TGN max_nodes={self.max_nodes}")

        self.memory.reset_state()
        x = snapshot.x.to(device)

        if snapshot.edge_events:
            for src, dst, time_index, raw_msg in snapshot.edge_events:
                src_t = torch.tensor([src], dtype=torch.long, device=device)
                dst_t = torch.tensor([dst], dtype=torch.long, device=device)
                t = torch.tensor([time_index], dtype=torch.long, device=device)
                msg = torch.tensor(np.asarray([raw_msg]), dtype=torch.float32, device=device)
                self.memory.update_state(src_t, dst_t, t, msg)

        n_id = torch.arange(snapshot.x.size(0), dtype=torch.long, device=device)
        memory, _ = self.memory(n_id)
        round_memory = memory[snapshot.round_node_index]
        round_node = self.node_proj(x[snapshot.round_node_index])
        logits = self.classifier(torch.cat([round_memory, round_node], dim=-1))
        prob = torch.sigmoid(logits).squeeze()
        return prob, memory


def train_static_model(train_pairs, global_stats, epochs: int = 8, lr: float = 1e-3, device: str = "cpu", shuffle: bool = True):
    model = create_static_model(train_pairs, global_stats, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        epoch_pairs = list(train_pairs)
        if shuffle:
            random.shuffle(epoch_pairs)
        for snapshot, label in iter_training_samples(epoch_pairs, global_stats):
            snap = GraphSnapshot(
                snapshot.x.to(device),
                snapshot.edge_index.to(device),
                snapshot.round_node_index,
                snapshot.round_node_indices,
                snapshot.metadata,
                snapshot.edge_events,
            )
            y = torch.tensor(label, dtype=torch.float32, device=device)
            prob, _ = model(snap)
            loss = F.binary_cross_entropy(prob, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_temporal_model(
    train_pairs,
    global_stats,
    epochs: int = 8,
    lr: float = 1e-3,
    device: str = "cpu",
    backbone: str = "gru",
    shuffle: bool = True,
):
    model = create_temporal_model(train_pairs, global_stats, device=device, backbone=backbone)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        epoch_pairs = list(train_pairs)
        if shuffle:
            random.shuffle(epoch_pairs)
        for snapshot, label in iter_training_samples(epoch_pairs, global_stats):
            if backbone == "tgn":
                model.memory.detach()
                model.memory.reset_state()
            y = torch.tensor(label, dtype=torch.float32, device=device)
            prob, _ = model.forward_snapshot(snapshot, device)
            loss = F.binary_cross_entropy(prob, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if backbone == "tgn":
                model.memory.detach()
    return model


def infer_motif(texts: list[str]):
    if not texts:
        return "none"
    seen = []
    for text in texts:
        reqs = request_types_from_text(text)
        if reqs:
            seen.extend(reqs)
    uniq = list(dict.fromkeys(seen))
    if "authority" in uniq and "payment" in uniq:
        return "authority_to_payment"
    if "urgency" in uniq and "payment" in uniq:
        return "urgency_to_payment"
    if "emotion" in uniq and "payment" in uniq:
        return "emotion_to_payment"
    if "credential" in uniq:
        return "credential_harvest"
    return "credibility_building"


def build_static_context(example: dict, texts: list[str], global_stats: dict, model: StaticGraphEncoder, device: str):
    snapshot = build_snapshot(example, texts, global_stats)
    snap = GraphSnapshot(
        snapshot.x.to(device),
        snapshot.edge_index.to(device),
        snapshot.round_node_index,
        snapshot.round_node_indices,
        snapshot.metadata,
        snapshot.edge_events,
    )
    with torch.no_grad():
        prob, _ = model(snap)
    current_text = texts[-1]
    reqs = request_types_from_text(current_text)
    return {
        "risk_score": round(float(prob.item()), 4),
        "risk_trend": "static",
        "round_index": len(texts),
        "risky_neighbors": reqs[:4],
        "category": example.get("category", "unknown"),
        "organization_signal": compact_organization_signal(example.get("role_bg", {}).get("Sender_company", "unknown")),
        "request_profile": reqs,
        "entity_summary": snapshot.metadata.get("entity_counts", {}),
        "global_reuse": {
            "organization_cases": global_stats["org_counts"].get(slugify(example.get("role_bg", {}).get("Sender_company", "unknown")), 1),
            "sender_role_cases": global_stats["role_counts"].get(slugify(example.get("role_bg", {}).get("Sender_occupation", "unknown")), 1),
        },
    }


def build_temporal_context(example: dict, texts: list[str], global_stats: dict, model, device: str):
    snapshot = build_snapshot(example, texts, global_stats)
    with torch.no_grad():
        current_prob, _ = model.forward_snapshot(snapshot, device)
        if len(texts) > 1:
            prev_snapshot = build_snapshot(example, texts[:-1], global_stats)
            prev_prob, _ = model.forward_snapshot(prev_snapshot, device)
            prev = float(prev_prob.item())
        else:
            prev = float(current_prob.item())
    current = float(current_prob.item())
    trend = "increasing" if current > prev + 1e-4 else "stable"
    reqs = request_types_from_text(texts[-1])
    return {
        "risk_score": round(current, 4),
        "risk_trend": trend,
        "round_index": len(texts),
        "risky_neighbors": reqs[:4],
        "temporal_motif": infer_motif(texts),
        "new_signal_this_round": reqs[0] if reqs else "none",
        "category": example.get("category", "unknown"),
        "organization_signal": compact_organization_signal(example.get("role_bg", {}).get("Sender_company", "unknown")),
        "entity_summary": snapshot.metadata.get("entity_counts", {}),
        "explanation_hint": f"risk moved from {prev:.3f} to {current:.3f}",
    }
