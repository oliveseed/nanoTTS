'''
Contains the model definition for both training and inference paths

Implements:
- standard rope
- flash attention kv cache for inference
- varlen attention for training
- swiglu or relu^2 mlp
- fused qkv
- gqa
- qk norm
'''

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm
from torch.nn.attention.varlen import varlen_attn
from flash_attn.flash_attn_interface import flash_attn_with_kvcache

#import ipdb

class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) implementation in PyTorch.
    """

    def __init__(
        self,
        embedding_dims: int,
        min_timescale: float = 1.0,
        max_timescale: float = 10000.0,
        dtype: torch.dtype = torch.float32,
        cache_max_len: int = 2048,
    ):
        super().__init__()
        if embedding_dims % 2 != 0:
            raise ValueError("Embedding dim must be even for RoPE.")
        self.embedding_dims = embedding_dims
        self.min_timescale = min_timescale
        self.max_timescale = max_timescale
        self.compute_dtype = dtype
        self.cache_max_len = cache_max_len

        half_embedding_dim = embedding_dims // 2
        fraction = (2.0 * torch.arange(0, half_embedding_dim)) / embedding_dims
        timescale = (self.min_timescale * (self.max_timescale / self.min_timescale) ** fraction).to(torch.float32)
        self.register_buffer("timescale", timescale, persistent=False)
        positions = torch.arange(cache_max_len, dtype=torch.float32).unsqueeze(-1)
        sinusoid_inp = positions / timescale
        self.register_buffer("sin_cache", torch.sin(sinusoid_inp), persistent=False)
        self.register_buffer("cos_cache", torch.cos(sinusoid_inp), persistent=False)

    def get_sin_cos(self, position: torch.Tensor):
        position = position.to(device=self.sin_cache.device, dtype=torch.long)
        sin = self.sin_cache[position]
        cos = self.cos_cache[position]
        return sin.unsqueeze(-2), cos.unsqueeze(-2)

    def forward(self, inputs: torch.Tensor, position: torch.Tensor):
        """Applies RoPE."""
        sin, cos = self.get_sin_cos(position)
        return self.apply_rope(inputs, sin, cos)

    def apply_rope(self, inputs: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        first_half, second_half = torch.chunk(inputs.to(torch.float32), 2, dim=-1)
        first_part = first_half * cos - second_half * sin
        second_part = second_half * cos + first_half * sin
        return torch.cat((first_part.to(self.compute_dtype), second_part.to(self.compute_dtype)), dim=-1)


class KVCache:
    def __init__(
        self, batch_size, max_seq_len, n_kv_heads, head_dim, device, dtype,
        k=None, v=None, seqlens=None,
    ):
        self.k = torch.empty(
            batch_size, max_seq_len, n_kv_heads, head_dim, device=device, dtype=dtype,
        ) if k is None else k

        self.v = torch.empty(
            batch_size, max_seq_len, n_kv_heads, head_dim, device=device, dtype=dtype,
        ) if v is None else v

        self.seqlens = torch.zeros(
            batch_size, dtype=torch.int32, device=device,
        ) if seqlens is None else seqlens


class MLP(nn.Module):#swiglu
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, 2*hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, in_dim, bias=False)
    def forward(self, x):
        x = self.in_proj(x)
        gate, up = x.chunk(2, dim=-1)
        return self.out_proj(up * F.silu(gate))

# class MLP(nn.Module):#relu^2
#     def __init__(self, in_dim, hidden_dim):
#         super().__init__()
#         self.in_proj = nn.Linear(in_dim, hidden_dim, bias=False)
#         self.out_proj = nn.Linear(hidden_dim, in_dim, bias=False)
#     def forward(self, x):
#         x = self.in_proj(x)
#         x = F.relu(x).square()
#         return self.out_proj(x)


class GQA(nn.Module):
    def __init__(
        self,
        q_embed_dim,
        kv_embed_dim,
        out_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        compute_dtype,
    ):
        super().__init__()
        self.q_embed_dim = q_embed_dim
        self.kv_embed_dim = kv_embed_dim
        self.out_embed_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads for GQA"
        self.num_gqa_groups = num_query_heads // num_kv_heads

        assert q_embed_dim == kv_embed_dim, "GQA fused qkv requires q_embed_dim == kv_embed_dim"
        self.q_proj_dim = num_query_heads * head_dim
        self.kv_proj_dim = num_kv_heads * head_dim
        self.qkv_proj = nn.Linear(q_embed_dim, self.q_proj_dim + (2 * self.kv_proj_dim), bias=False)
        self.out_proj = nn.Linear(num_query_heads * head_dim, out_embed_dim, bias=False)

        # explicitly set bf16 because https://github.com/pytorch/pytorch/issues/167308
        # or else it won't use fused rmsnorm and will be slower
        # TODO: some parts still have fp32 rmsnorm
        self.norm_q = RMSNorm(head_dim, eps=1e-5, dtype=torch.bfloat16)
        self.norm_k = RMSNorm(head_dim, eps=1e-5, dtype=torch.bfloat16)
        
        self.rotary_emb = RotaryEmbedding(
            embedding_dims=head_dim,
            min_timescale=1.0,
            max_timescale=10000.0,
            dtype=torch.bfloat16,
        )

    def _project_qkv(self, x: torch.Tensor):
        qkv = self.qkv_proj(x)
        q, k, v = torch.split(qkv, [self.q_proj_dim, self.kv_proj_dim, self.kv_proj_dim], dim=-1)
        return q, k, v

    def forward_inference(self, x, pos, attn_mask=None, cache=None):#, prefill=False):
        batch_size, seq_len, _ = x.shape

        q, k, v = self._project_qkv(x)
        q = q.view(batch_size, seq_len, self.num_query_heads, self.head_dim)   # (B, T, N, H)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = self.rotary_emb(q, pos)
        k = self.rotary_emb(k, pos)

        if cache is not None:
            # decoder self-attn requires kvcache at inference
            # q, k, v: (B, S, N, H)
            attn_output = flash_attn_with_kvcache(
                q=q,
                k_cache=cache.k,
                v_cache=cache.v,
                k=k,
                v=v,
                cache_seqlens=cache.seqlens,
                causal=True,
            )
            cache.seqlens += k.shape[1] # should be same as += 1
        else:
            # encoder self-attn does not use kvcache
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            # q, k, v: (B, N, S, H)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                # scale=1.0,
                enable_gqa=self.num_gqa_groups > 1,
            )   # (B, N, S, H)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.q_embed_dim)
        output = self.out_proj(attn_output)

        return output

    def forward(self, x, pos, cu_seq, max_len, is_causal):
        # orig_dtype = x.dtype

        q, k, v = self._project_qkv(x)
        q = q.view(-1, self.num_query_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = self.rotary_emb(q, pos)
        k = self.rotary_emb(k, pos)

        window_size = (-1, 0) if is_causal else (-1, -1)
        attn_output = varlen_attn(
            query=q,
            key=k,
            value=v,
            cu_seq_q=cu_seq,
            cu_seq_k=cu_seq,
            max_q=max_len,
            max_k=max_len,
            window_size=window_size,
        )

        attn_output = attn_output.contiguous().view(-1, self.q_embed_dim)
        output = self.out_proj(attn_output)

        return output#.to(orig_dtype)


class GQCA(nn.Module):
    def __init__(
        self,
        q_embed_dim,
        kv_embed_dim,
        out_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        compute_dtype,
    ):
        super().__init__()
        self.q_embed_dim = q_embed_dim
        self.kv_embed_dim = kv_embed_dim
        self.out_embed_dim = out_embed_dim if out_embed_dim is not None else q_embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads for GQA"
        self.num_gqa_groups = num_query_heads // num_kv_heads

        self.q_proj = nn.Linear(q_embed_dim, num_query_heads * head_dim, bias=False)
        self.kv_proj_dim = num_kv_heads * head_dim
        self.kv_proj = nn.Linear(kv_embed_dim, 2 * self.kv_proj_dim, bias=False)
        self.out_proj = nn.Linear(num_query_heads * head_dim, out_embed_dim, bias=False)
        
        self.norm_q = RMSNorm(head_dim, eps=1e-5, dtype=torch.bfloat16)
        self.norm_k = RMSNorm(head_dim, eps=1e-5, dtype=torch.bfloat16)

        self.rotary_emb = RotaryEmbedding(
            embedding_dims=head_dim,
            min_timescale=1.0,
            max_timescale=10000.0,
            dtype=torch.bfloat16,
        )

    def _project_kv(self, xkv: torch.Tensor):
        kv = self.kv_proj(xkv)
        k, v = torch.split(kv, [self.kv_proj_dim, self.kv_proj_dim], dim=-1)
        return k, v

    def forward_inference(self, xq, q_pos, cache):
        batch_size, seq_len_q, _ = xq.shape

        q = self.q_proj(xq).view(batch_size, seq_len_q, self.num_query_heads, self.head_dim)
        q = self.norm_q(q)
        q = self.rotary_emb(q, q_pos)
        # q: (B, T, N, H)
        attn_output = flash_attn_with_kvcache(
            q=q,
            k_cache=cache.k,
            v_cache=cache.v,
            cache_seqlens=cache.seqlens,
            causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.q_embed_dim)   # cross attn uses query shapes
        output = self.out_proj(attn_output)

        return output

    def forward(
        self, xq, xkv, q_pos, kv_pos,
        cu_seq_q, cu_seq_k, max_q, max_k,
    ):
        # orig_dtype = xq.dtype

        q = self.q_proj(xq).view(-1, self.num_query_heads, self.head_dim)
        k, v = self._project_kv(xkv)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        q = self.norm_q(q)
        k = self.norm_k(k)

        q = self.rotary_emb(q, q_pos)
        k = self.rotary_emb(k, kv_pos)

        attn_output = varlen_attn(
            query=q,
            key=k,
            value=v,
            cu_seq_q=cu_seq_q,
            cu_seq_k=cu_seq_k,
            max_q=max_q,
            max_k=max_k,
            window_size=(-1, -1),   # full attention over encoder output
        )

        attn_output = attn_output.contiguous().view(-1, self.q_embed_dim)   # cross attn uses query shapes
        output = self.out_proj(attn_output)

        return output#.to(orig_dtype)


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        expand_dim,
        norm_eps,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        self.pre_norm = RMSNorm(embed_dim, eps=norm_eps, dtype=torch.float32)
        self.sa = GQA(
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            out_embed_dim=embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            compute_dtype=torch.float32,
        )

        self.post_norm = RMSNorm(embed_dim, eps=norm_eps, dtype=torch.float32)
        self.mlp = MLP(
            in_dim=embed_dim,
            hidden_dim=expand_dim,
        )

    def forward_inference(self, x, pos, attn_mask=None):#, prefill=False,):
        residual = x
        x_norm = self.pre_norm(x)
        attn_out = self.sa.forward_inference(
            x=x_norm,
            pos=pos,
            attn_mask=attn_mask,
            cache=None,
            # prefill=prefill,
        )
        x = residual + attn_out
        residual = x
        x_norm = self.post_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out
        return x

    def forward(self, x, pos, cu_seq, max_len):
        residual = x
        x_norm = self.pre_norm(x)
        attn_out = self.sa(
            x=x_norm,
            pos=pos,
            cu_seq=cu_seq,
            max_len=max_len,
            is_causal=False,
        )
        x = residual + attn_out
        residual = x
        x_norm = self.post_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        expand_dim,
        norm_eps,
        vocab_size,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps
        self.vocab_size = vocab_size

        # TODO: find out why state has to be loaded here for it to work
        # emb_state = torch.load('../elm_emb.pth', map_location='cpu')
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # self.embedding.load_state_dict(emb_state)

        self.layers = nn.ModuleList([
            EncoderLayer(
                embed_dim=embed_dim,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                expand_dim=expand_dim,
                norm_eps=norm_eps,
            )
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim, eps=norm_eps, dtype=torch.float32)

    def forward_inference(self, x_ids, pos, attn_mask=None):#, prefill=False,):
        x = self.embedding(x_ids)
        for layer in self.layers:
            x = layer.forward_inference(
                x=x,
                pos=pos,
                attn_mask=attn_mask,
                # prefill=prefill,
            )
        x = self.norm(x)
        return x

    def forward(self, x_ids, pos, cu_seq, max_len):
        x = self.embedding(x_ids)
        for layer in self.layers:
            x = layer(
                x=x,
                pos=pos,
                cu_seq=cu_seq,
                max_len=max_len,
            )
        x = self.norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        enc_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        ca_num_query_heads,
        ca_num_kv_heads,
        ca_head_dim,
        expand_dim,
        norm_eps=1e-5,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.norm_eps = norm_eps

        self.pre_sa_norm = RMSNorm(embed_dim, eps=norm_eps, dtype=torch.float32)
        self.pre_ca_norm = RMSNorm(embed_dim, eps=norm_eps, dtype=torch.float32)
        self.pre_mlp_norm = RMSNorm(embed_dim, eps=norm_eps, dtype=torch.float32)

        self.sa = GQA(
            q_embed_dim=embed_dim,
            kv_embed_dim=embed_dim,
            out_embed_dim=embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            compute_dtype=torch.float32,
        )
        self.ca = GQCA(
            q_embed_dim=embed_dim,
            kv_embed_dim=enc_embed_dim,
            out_embed_dim=embed_dim,
            num_query_heads=ca_num_query_heads,
            num_kv_heads=ca_num_kv_heads,
            head_dim=ca_head_dim,
            compute_dtype=torch.float32,
        )
        self.mlp = MLP(
            in_dim=embed_dim,
            hidden_dim=expand_dim,
        )

    def forward_inference(self, x, enc_kv_pos, dec_pos, sa_cache, ca_cache):
        residual = x
        x_norm = self.pre_sa_norm(x)

        sa_out = self.sa.forward_inference(
            x=x_norm, pos=dec_pos, cache=sa_cache, # prefill=prefill,
        )
        x = residual + sa_out

        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out = self.ca.forward_inference(
            xq=x_norm, q_pos=dec_pos, cache=ca_cache,
        )
        x = residual + ca_out

        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x

    def forward(
        self, x, enc_out, 
        enc_kv_pos, enc_cu_seq, enc_max_len,
        dec_q_pos, dec_cu_seq, dec_max_len,
    ):
        residual = x
        x_norm = self.pre_sa_norm(x)

        sa_out = self.sa(
            x=x_norm,
            pos=dec_q_pos,
            cu_seq=dec_cu_seq,
            max_len=dec_max_len,
            is_causal=True,
        )
        x = residual + sa_out

        residual = x
        x_norm = self.pre_ca_norm(x)
        ca_out = self.ca(
            xq=x_norm,
            xkv=enc_out,
            q_pos=dec_q_pos,
            kv_pos=enc_kv_pos,
            cu_seq_q=dec_cu_seq,
            cu_seq_k=enc_cu_seq,
            max_q=dec_max_len,
            max_k=enc_max_len,
        )
        x = residual + ca_out

        residual = x
        x_norm = self.pre_mlp_norm(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        num_layers,
        embed_dim,
        enc_embed_dim,
        num_query_heads,
        num_kv_heads,
        head_dim,
        ca_num_query_heads,
        ca_num_kv_heads,
        ca_head_dim,
        expand_dim,
        norm_eps,
        vocab_size,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        assert embed_dim // num_query_heads == head_dim

        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.layers = nn.ModuleList([
            DecoderLayer(
                embed_dim=embed_dim,
                enc_embed_dim=enc_embed_dim,
                num_query_heads=num_query_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                ca_num_query_heads=ca_num_query_heads,
                ca_num_kv_heads=ca_num_kv_heads,
                ca_head_dim=ca_head_dim,
                expand_dim=expand_dim,
            )
            for _ in range(num_layers)
        ])

        self.norm = RMSNorm(embed_dim, eps=norm_eps, dtype=torch.float32)

        self.logits_proj = nn.Linear(embed_dim, vocab_size, bias=False)
        # self.logits_proj.weight = self.embeddings.weight

    def precompute_cross_attn_kv(
        self,
        max_len: int,
        enc_out: torch.Tensor,  # (B, S, E)
        enc_lens: torch.Tensor,
        src_positions: torch.Tensor,
    ) -> list[KVCache]:
        """
        Computes the Key and Value tensors for cross-attention for each layer from the encoder output.
        """
        per_layer_kv_cache: list[KVCache] = []

        batch_size, seq_len, _ = enc_out.shape

        for layer in self.layers:
            cross_attn_module = layer.ca
            k, v = cross_attn_module._project_kv(enc_out)
            k = k.view(batch_size, seq_len, cross_attn_module.num_kv_heads, cross_attn_module.head_dim)
            v = v.view(batch_size, seq_len, cross_attn_module.num_kv_heads, cross_attn_module.head_dim)

            k = cross_attn_module.norm_k(k)
            k = cross_attn_module.rotary_emb(k, position=src_positions)

            per_layer_kv_cache.append(
                KVCache(
                    batch_size=batch_size,
                    max_seq_len=max_len,
                    n_kv_heads=cross_attn_module.num_kv_heads,
                    head_dim=cross_attn_module.head_dim,
                    device=k.device,
                    dtype=torch.bfloat16,
                    k=k,
                    v=v,
                    seqlens=enc_lens,
                )
            )

        return per_layer_kv_cache

    def decode_step(
        self,
        src_pos,
        tgt_ids,
        tgt_pos,
        sa_cache,
        ca_cache,
    ):
        '''
        only used at inference.
        caches should not be None
        '''

        x = self.embeddings(tgt_ids)

        for i, layer in enumerate(self.layers):
            x = layer.forward_inference(
                x=x,
                enc_kv_pos=src_pos,
                dec_pos=tgt_pos,
                sa_cache=sa_cache[i],
                ca_cache=ca_cache[i],
            )
        x = self.norm(x)
        logits = self.logits_proj(x)
        return logits
  
    def forward(
        self,
        enc_out,
        src_pos,
        src_cu_seqlens,
        src_max_seqlen,
        tgt_ids,
        tgt_pos,
        tgt_cu_seqlens,
        tgt_max_seqlen,
        loss_reduction='mean',
    ):
        '''
        only used at training.
        caches are set to None and it should still work
        '''

        x = self.embeddings(tgt_ids)

        for i, layer in enumerate(self.layers):
            x = layer(
                x=x,
                enc_out=enc_out,
                enc_kv_pos=src_pos,
                enc_cu_seq=src_cu_seqlens,
                enc_max_len=src_max_seqlen,
                dec_q_pos=tgt_pos,
                # dec_kv_pos=tgt_pos,
                dec_cu_seq=tgt_cu_seqlens,
                dec_max_len=tgt_max_seqlen,
            )
        x = self.norm(x)
        
        logits = self.logits_proj(x) # (BT, V)
        # logits = logits.float()
        # softcap = 15
        # logits = softcap * torch.tanh(logits / softcap)

        ce_loss = F.cross_entropy(logits[:-1], tgt_ids[1:], reduction=loss_reduction)

        return ce_loss


class TTSModel(nn.Module):
    def __init__(
        self,
        enc_num_layers,
        enc_embed_dim,
        enc_hidden_dim,
        enc_head_dim,
        dec_num_layers,
        dec_embed_dim,
        dec_hidden_dim,
        dec_head_dim,
        num_query_heads,
        num_kv_heads,
        ca_num_query_heads,
        ca_num_kv_heads,
        ca_head_dim,
        enc_vocab_size,
        dec_vocab_size,
    ):
        super().__init__()
        self.encoder = Encoder(
            num_layers=enc_num_layers,
            embed_dim=enc_embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=enc_head_dim,
            expand_dim=enc_hidden_dim,
            norm_eps=1e-5,
            vocab_size=enc_vocab_size,
        )
        self.decoder = Decoder(
            num_layers=dec_num_layers,
            embed_dim=dec_embed_dim,
            enc_embed_dim=enc_embed_dim,
            num_query_heads=num_query_heads,
            num_kv_heads=num_kv_heads,
            head_dim=dec_head_dim,
            ca_num_query_heads=ca_num_query_heads,
            ca_num_kv_heads=ca_num_kv_heads,
            ca_head_dim=ca_head_dim,
            expand_dim=dec_hidden_dim,
            norm_eps=1e-5,
            vocab_size=dec_vocab_size,
        )

    def _init_weights(self):
        print("initializing weights\n")
        nn.init.normal_(self.decoder.embeddings.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.decoder.logits_proj.weight, mean=0.0, std=0.001)
        
        enc_embed_dim = self.encoder.embed_dim
        dec_embed_dim = self.decoder.embed_dim
        se = 3**0.5 * enc_embed_dim**-0.5
        sd = 3**0.5 * dec_embed_dim**-0.5
        for layer in self.encoder.layers:
            # self attn
            nn.init.uniform_(layer.sa.qkv_proj.weight, -se, se)
            nn.init.zeros_(layer.sa.out_proj.weight)
            # mlp
            nn.init.zeros_(layer.mlp.out_proj.weight)
        for layer in self.decoder.layers:
            # self attn
            nn.init.uniform_(layer.sa.qkv_proj.weight, -sd, sd)
            nn.init.zeros_(layer.sa.out_proj.weight)
            # cross attn
            nn.init.uniform_(layer.ca.q_proj.weight, -sd, sd)
            nn.init.uniform_(layer.ca.kv_proj.weight, -sd, sd)
            nn.init.zeros_(layer.ca.out_proj.weight)
            # mlp
            nn.init.zeros_(layer.mlp.out_proj.weight)

    def forward(
        self,
        src_ids,
        src_pos,
        src_cu_seqlens,
        src_max_seqlen,
        tgt_ids,
        tgt_pos,
        tgt_cu_seqlens,
        tgt_max_seqlen,
        loss_reduction='mean',
    ):
        '''
        only used at training.
        caches are set to None and it should still work
        '''

        enc_out = self.encoder(
            src_ids,
            src_pos,
            src_cu_seqlens,
            src_max_seqlen,
        )

        dec_out = self.decoder(
            enc_out=enc_out,
            src_pos=src_pos,
            src_cu_seqlens=src_cu_seqlens,
            src_max_seqlen=src_max_seqlen,
            tgt_ids=tgt_ids,
            tgt_pos=tgt_pos,
            tgt_cu_seqlens=tgt_cu_seqlens,
            tgt_max_seqlen=tgt_max_seqlen,
            loss_reduction=loss_reduction,
        )
        return dec_out
