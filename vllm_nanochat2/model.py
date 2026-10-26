# coding=utf-8
"""Implementação nativa para o vLLM (testada contra a API das versões 0.26 e 0.28) dos modelos
nanochat convertidos com convert_nanochat_to_hf.py (arquitectura "Nanochat2ForCausalLM").

Só uma GPU (sem tensor/pipeline parallel). Os nomes dos parâmetros são os mesmos do
checkpoint HF (que são os do nanochat), por isso os pesos carregam sem mapeamento.

O problema do "smear": cada token precisa do embedding do token anterior, mas o vLLM só dá ao
modelo os tokens novos de cada pedido. Solução: uma camada de atenção auxiliar (sem pesos) que
guarda na cache do vLLM o id de cada token (em 4 nibbles) e usa uma codificação posicional
fixa que faz cada query atender, com peso ~1, apenas à posição anterior. Assim o próprio vLLM
guarda e devolve o token anterior, com batching, prefill por blocos e cache de prefixos.
A camada tem a mesma forma e janela que as camadas S, para partilhar o mesmo grupo de cache.
"""

import math
from collections.abc import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    # x: (N, H, D); cos/sin: (N, 1, D/2). Roda por -theta, como o nanochat.
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], dim=-1)


class Linear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx, vllm_config, prefix):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.num_attention_heads
        self.n_kv_head = config.num_key_value_heads
        self.n_embd = config.hidden_size
        self.head_dim = self.n_embd // self.n_head
        self.qk_scale = config.qk_scale
        self.ve_gate_channels = config.ve_gate_channels
        self.ve_gate_scale = config.ve_gate_scale
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        if has_ve(layer_idx, config.num_hidden_layers):
            self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
        else:
            self.ve_gate = None

        # nanochat: a query i vê a chave j se (i - j) <= janela. No vLLM, sliding_window = janela + 1.
        # Janelas >= contexto máximo equivalem a atenção completa.
        window = config.window_sizes[layer_idx]
        sliding = window + 1 if window + 1 < config.max_position_embeddings else None
        self.attn = Attention(
            self.n_head,
            self.head_dim,
            self.head_dim**-0.5,
            num_kv_heads=self.n_kv_head,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=sliding,
            prefix=f"{prefix}.attn",
        )

    def forward(self, x, ve, cos, sin):
        N = x.shape[0]
        q = self.c_q(x).view(N, self.n_head, self.head_dim)
        k = self.c_k(x).view(N, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(N, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(N, self.n_kv_head, self.head_dim)
            gate = self.ve_gate_scale * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q = norm(q) * self.qk_scale
        k = norm(k) * self.qk_scale
        y = self.attn(q.reshape(N, -1), k.reshape(N, -1), v.reshape(N, -1))
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.hidden_size, 4 * config.hidden_size, bias=False)
        self.c_proj = Linear(4 * config.hidden_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(nn.Module):
    def __init__(self, config, layer_idx, vllm_config, prefix):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx, vllm_config, prefix=f"{prefix}.attn")
        self.mlp = MLP(config)

    def forward(self, x, ve, cos, sin):
        x = x + self.attn(norm(x), ve, cos, sin)
        x = x + self.mlp(norm(x))
        return x


class PrevTokenAttention(nn.Module):
    """Devolve o id do token na posição anterior (ver docstring do módulo). Não tem pesos."""

    NIBBLES = 4  # 4 x 4 bits = ids até 65535; inteiros <= 15 são exactos até em fp8

    def __init__(self, config, vllm_config, prefix):
        super().__init__()
        self.n_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        half = self.head_dim // 2
        # frequências escolhidas para que, numa janela de 513, a posição anterior tenha um score
        # pelo menos ~48 acima de qualquer outra (peso das restantes < 1e-18)
        freqs = math.pi * torch.sqrt((torch.arange(half, dtype=torch.float32) + 0.5) / half)
        self.register_buffer("freqs", freqs, persistent=False)
        short = [w for w in config.window_sizes if w + 1 < config.max_position_embeddings]
        window = (min(short) + 1) if short else 513  # partilha o grupo de cache das camadas S
        self.attn = Attention(
            self.n_heads,
            self.head_dim,
            1.0,
            num_kv_heads=self.n_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=window,
            prefix=f"{prefix}.attn",
        )
    def forward(self, input_ids, positions, dtype):
        N = input_ids.shape[0]
        pos = positions.to(torch.float32)
        ang_k = pos[:, None] * self.freqs
        ang_q = (pos - 1)[:, None] * self.freqs
        k0 = torch.cat([ang_k.cos(), ang_k.sin()], dim=-1)
        q0 = torch.cat([ang_q.cos(), ang_q.sin()], dim=-1)
        ids = input_ids.to(torch.int64)
        nib = torch.stack([(ids >> (4 * i)) & 15 for i in range(self.NIBBLES)], dim=-1).to(torch.float32)
        v0 = torch.cat([nib, nib.new_zeros(N, self.head_dim - self.NIBBLES)], dim=-1)
        pad = q0.new_zeros(N, (self.n_heads - 1) * self.head_dim)
        q = torch.cat([q0, pad], dim=-1).to(dtype)
        k = torch.cat([k0, pad], dim=-1).to(dtype)
        v = torch.cat([v0, pad], dim=-1).to(dtype)
        out = self.attn(q, k, v)[:, : self.NIBBLES].float().round().clamp(0, 15).to(torch.int64)
        prev = out[:, 0]
        for i in range(1, self.NIBBLES):
            prev = prev + (out[:, i] << (4 * i))
        return prev


class Nanochat2ForCausalLM(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        if vllm_config.parallel_config.tensor_parallel_size != 1 or vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("Nanochat2ForCausalLM: só suporta uma GPU (tensor/pipeline parallel = 1).")
        if not hasattr(config, "window_sizes"):
            raise ValueError("config.json sem window_sizes: o modelo foi convertido com convert_nanochat_to_hf.py?")
        n_layer = config.num_hidden_layers
        V = config.padded_vocab_size
        C = config.hidden_size
        kv_dim = config.num_key_value_heads * (C // config.num_attention_heads)

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(V, C),
                "h": nn.ModuleList(
                    [Block(config, i, vllm_config, prefix=f"transformer.h.{i}") for i in range(n_layer)]
                ),
            }
        )
        self.lm_head = Linear(C, V, bias=False)
        # escalares em fp32, como no nanochat
        self.resid_lambdas = nn.Parameter(torch.ones(n_layer, dtype=torch.float32))
        self.x0_lambdas = nn.Parameter(torch.zeros(n_layer, dtype=torch.float32))
        self.smear_gate = Linear(config.smear_channels, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1, dtype=torch.float32))
        self.value_embeds = nn.ModuleDict(
            {str(i): nn.Embedding(V, kv_dim) for i in range(n_layer) if has_ve(i, n_layer)}
        )
        # o nome tem de conter um único inteiro, diferente dos das outras camadas
        self.prev_token = PrevTokenAttention(config, vllm_config, prefix=f"prev_token.{n_layer}")

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.wte(input_ids)

    def _rotary(self, positions, dtype):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=positions.device)
        inv_freq = 1.0 / (self.config.rope_base ** (channel_range / head_dim))
        freqs = positions.to(torch.float32)[:, None] * inv_freq
        return freqs.cos().to(dtype)[:, None, :], freqs.sin().to(dtype)[:, None, :]

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("Nanochat2ForCausalLM precisa de input_ids (prompt embeds não suportados).")
        cfg = self.config
        x = norm(self.transformer.wte(input_ids))
        dtype = x.dtype

        # smear com o token anterior (vindo da cache através da atenção auxiliar)
        prev_ids = self.prev_token(input_ids, positions, dtype).clamp(max=cfg.padded_vocab_size - 1)
        has_prev = (positions > 0).to(dtype)[:, None]
        x_prev = norm(self.transformer.wte(prev_ids)) * has_prev
        gate = self.smear_lambda.to(dtype) * torch.sigmoid(self.smear_gate(x[..., : cfg.smear_channels]))
        x = x + gate * x_prev

        cos, sin = self._rotary(positions, dtype)
        x0 = x
        backout_layer = cfg.num_hidden_layers // 2
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](input_ids).to(dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos, sin)
            if i == backout_layer:
                x_backout = x
        if x_backout is not None:
            x = x - self.backout_lambda.to(dtype) * x_backout
        return norm(x)

    def compute_logits(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        logits = self.lm_head(hidden_states)[..., : self.config.vocab_size].float()
        softcap = self.config.logit_softcap
        return softcap * torch.tanh(logits / softcap)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded = set()
        for name, w in weights:
            name = name.removeprefix("_orig_mod.")
            if name not in params:
                raise ValueError(f"Peso inesperado no checkpoint: {name}")
            param = params[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, w.to(param.dtype))
            loaded.add(name)
        # parâmetros internos das camadas Attention do vLLM (escalas de quantização) não vêm do checkpoint
        internal = {
            f"{mod_name}.{p}"
            for mod_name, mod in self.named_modules()
            if isinstance(mod, Attention)
            for p, _ in mod.named_parameters()
        }
        missing = set(params) - loaded - internal
        if missing:
            raise ValueError(f"Pesos em falta no checkpoint: {sorted(missing)}")
        return loaded
