from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GPT2Config:
    vocab_size: int = 32000
    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    bias: bool = True


GPT2_PRESETS = {
    "gpt2": GPT2Config(n_layer=12, n_head=12, n_embd=768),
    "gpt2-medium": GPT2Config(n_layer=24, n_head=16, n_embd=1024),
    "gpt2-large": GPT2Config(n_layer=36, n_head=20, n_embd=1280),
    "gpt2-xl": GPT2Config(n_layer=48, n_head=25, n_embd=1600),
    "gpt2-tiny": GPT2Config(n_layer=4, n_head=4, n_embd=128, block_size=128),
}


class Parameter:
    def __init__(self, data: np.ndarray):
        self.data = data.astype(np.float32)
        self.grad = np.zeros_like(self.data)


class Linear:
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        self.w = Parameter(np.random.randn(in_dim, out_dim) * 0.02)
        self.b = Parameter(np.zeros(out_dim)) if bias else None
        self._x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        y = x @ self.w.data
        if self.b is not None:
            y = y + self.b.data
        return y

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        x2 = self._x.reshape(-1, self._x.shape[-1])
        g2 = grad_out.reshape(-1, grad_out.shape[-1])
        self.w.grad += x2.T @ g2
        if self.b is not None:
            self.b.grad += g2.sum(axis=0)
        return grad_out @ self.w.data.T

    def parameters(self) -> list[Parameter]:
        return [self.w] + ([self.b] if self.b is not None else [])


class Embedding:
    def __init__(self, n: int, d: int):
        self.w = Parameter(np.random.randn(n, d) * 0.02)
        self._idx = None

    def forward(self, idx: np.ndarray) -> np.ndarray:
        self._idx = idx
        return self.w.data[idx]

    def backward(self, grad_out: np.ndarray) -> None:
        np.add.at(self.w.grad, self._idx, grad_out)

    def parameters(self) -> list[Parameter]:
        return [self.w]


class LayerNorm:
    def __init__(self, d: int, eps: float = 1e-5):
        self.g = Parameter(np.ones(d))
        self.b = Parameter(np.zeros(d))
        self.eps = eps
        self._x = self._xhat = self._inv = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        m = x.mean(axis=-1, keepdims=True)
        v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(v + self.eps)
        xhat = (x - m) * inv
        self._x, self._xhat, self._inv = x, xhat, inv
        return self.g.data * xhat + self.b.data

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        d = grad_out.shape[-1]
        self.g.grad += (grad_out * self._xhat).sum(axis=(0, 1))
        self.b.grad += grad_out.sum(axis=(0, 1))
        dxhat = grad_out * self.g.data
        s1 = dxhat.sum(axis=-1, keepdims=True)
        s2 = (dxhat * self._xhat).sum(axis=-1, keepdims=True)
        return (1.0 / d) * self._inv * (d * dxhat - s1 - self._xhat * s2)

    def parameters(self) -> list[Parameter]:
        return [self.g, self.b]


def gelu(x: np.ndarray) -> np.ndarray:
    a = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(a * (x + 0.044715 * x**3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    a = np.sqrt(2.0 / np.pi)
    u = a * (x + 0.044715 * x**3)
    t = np.tanh(u)
    sech2 = 1.0 - t**2
    return 0.5 * (1.0 + t) + 0.5 * x * sech2 * a * (1.0 + 3.0 * 0.044715 * x**2)


class MLP:
    def __init__(self, d: int, bias: bool):
        self.fc = Linear(d, 4 * d, bias=bias)
        self.proj = Linear(4 * d, d, bias=bias)
        self._h = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = self.fc.forward(x)
        self._h = h
        return self.proj.forward(gelu(h))

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        gh = self.proj.backward(grad_out)
        gh = gh * gelu_grad(self._h)
        return self.fc.backward(gh)

    def parameters(self) -> list[Parameter]:
        return self.fc.parameters() + self.proj.parameters()


class CausalSelfAttention:
    def __init__(self, config: GPT2Config):
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.c_attn = Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = Linear(config.n_embd, config.n_embd, bias=config.bias)
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        b, t, c = x.shape
        qkv = self.c_attn.forward(x)
        q, k, v = np.split(qkv, 3, axis=2)
        q = q.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(self.head_dim)
        mask = np.tril(np.ones((t, t), dtype=np.float32))[None, None, :, :]
        scores = np.where(mask == 1.0, scores, -1e9)
        maxs = scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores - maxs)
        probs = exp / exp.sum(axis=-1, keepdims=True)

        y = probs @ v
        y2 = y.transpose(0, 2, 1, 3).reshape(b, t, c)
        out = self.c_proj.forward(y2)
        self._cache = (q, k, v, probs, y, b, t, c)
        return out

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        q, k, v, probs, y, b, t, c = self._cache
        gy2 = self.c_proj.backward(grad_out)
        gy = gy2.reshape(b, t, self.n_head, self.head_dim).transpose(0, 2, 1, 3)

        gprobs = gy @ v.transpose(0, 1, 3, 2)
        gv = probs.transpose(0, 1, 3, 2) @ gy

        s = (gprobs * probs).sum(axis=-1, keepdims=True)
        gscores = probs * (gprobs - s)
        gscores /= math.sqrt(self.head_dim)

        gq = gscores @ k
        gk = gscores.transpose(0, 1, 3, 2) @ q

        gq = gq.transpose(0, 2, 1, 3).reshape(b, t, c)
        gk = gk.transpose(0, 2, 1, 3).reshape(b, t, c)
        gv = gv.transpose(0, 2, 1, 3).reshape(b, t, c)

        gqkv = np.concatenate([gq, gk, gv], axis=2)
        return self.c_attn.backward(gqkv)

    def parameters(self) -> list[Parameter]:
        return self.c_attn.parameters() + self.c_proj.parameters()


class Block:
    def __init__(self, config: GPT2Config):
        self.ln_1 = LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd)
        self.mlp = MLP(config.n_embd, bias=config.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        a = self.attn.forward(self.ln_1.forward(x))
        self._x2 = x + a
        m = self.mlp.forward(self.ln_2.forward(self._x2))
        return self._x2 + m

    def backward(self, grad_out: np.ndarray) -> np.ndarray:
        g_mlp_in = self.mlp.backward(grad_out)
        g_x2 = grad_out + self.ln_2.backward(g_mlp_in)
        g_attn_in = self.attn.backward(g_x2)
        return g_x2 + self.ln_1.backward(g_attn_in)

    def parameters(self) -> list[Parameter]:
        return self.ln_1.parameters() + self.attn.parameters() + self.ln_2.parameters() + self.mlp.parameters()


class GPT2LMHeadModel:
    def __init__(self, config: GPT2Config):
        self.config = config
        self.wte = Embedding(config.vocab_size, config.n_embd)
        self.wpe = Embedding(config.block_size, config.n_embd)
        self.blocks = [Block(config) for _ in range(config.n_layer)]
        self.ln_f = LayerNorm(config.n_embd)

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad.fill(0.0)

    def parameters(self) -> list[Parameter]:
        params = self.wte.parameters() + self.wpe.parameters()
        for b in self.blocks:
            params += b.parameters()
        params += self.ln_f.parameters()
        return params

    def num_parameters(self) -> int:
        return sum(p.data.size for p in self.parameters())

    def forward(self, idx: np.ndarray) -> np.ndarray:
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError("input length exceeds block_size")
        pos = np.broadcast_to(np.arange(t)[None, :], (b, t))
        x = self.wte.forward(idx) + self.wpe.forward(pos)
        for block in self.blocks:
            x = block.forward(x)
        x = self.ln_f.forward(x)
        self._x_final = x
        return x @ self.wte.w.data.T

    def loss_and_backward(self, idx: np.ndarray, targets: np.ndarray) -> float:
        logits = self.forward(idx)
        b, t, v = logits.shape
        flat = logits.reshape(-1, v)
        tgt = targets.reshape(-1)

        maxs = flat.max(axis=-1, keepdims=True)
        exp = np.exp(flat - maxs)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        n = tgt.shape[0]
        loss = -np.log(probs[np.arange(n), tgt] + 1e-12).mean()

        gflat = probs
        gflat[np.arange(n), tgt] -= 1.0
        gflat /= n
        glogits = gflat.reshape(b, t, v)

        self.wte.w.grad += glogits.reshape(-1, v).T @ self._x_final.reshape(-1, self.config.n_embd)
        gx = glogits @ self.wte.w.data

        gx = self.ln_f.backward(gx)
        for block in reversed(self.blocks):
            gx = block.backward(gx)

        self.wte.backward(gx)
        self.wpe.backward(gx)
        return float(loss)

    def generate(self, idx: np.ndarray, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = 50) -> np.ndarray:
        out = idx.copy()
        for _ in range(max_new_tokens):
            cond = out[:, -self.config.block_size :]
            logits = self.forward(cond)[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None and int(top_k) > 0:
                k = min(int(top_k), logits.shape[-1])
                kth = np.partition(logits, -k, axis=-1)[:, -k][:, None]
                logits = np.where(logits < kth, -1e9, logits)
            probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
            probs /= probs.sum(axis=-1, keepdims=True)
            next_ids = np.array([np.random.choice(probs.shape[1], p=p) for p in probs], dtype=np.int64)[:, None]
            out = np.concatenate([out, next_ids], axis=1)
        return out


class AdamW:
    def __init__(self, params: list[Parameter], lr: float = 3e-4, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.01):
        self.params = params
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self) -> None:
        self.t += 1
        for i, p in enumerate(self.params):
            g = p.grad + self.wd * p.data
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            mhat = self.m[i] / (1 - self.b1**self.t)
            vhat = self.v[i] / (1 - self.b2**self.t)
            p.data -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
