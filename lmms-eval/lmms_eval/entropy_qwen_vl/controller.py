import math
import torch
import torch.nn.functional as F

def _visual_entropy(probs, vmask4d):
    p_vis = probs * vmask4d
    mass = p_vis.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    p_norm = p_vis / mass
    H_q = -(p_norm * p_norm.clamp_min(1e-12).log()).sum(dim=-1)
    H_vis = H_q.mean(dim=-1)
    return H_vis


class KVEntropyControllerLite:
    def __init__(self, attn_module, num_heads, head_dim, num_kv_heads,
                 layer_key, direct, idxs, d_a, d_b):
        self.attn = attn_module
        self.layer_key = layer_key
        self.direct = direct
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_kv_heads = num_kv_heads
        self.kv_head_dim = head_dim
        self.idxs = idxs
        self.d_a = d_a
        self.d_b = d_b

    def _build_visual_mask_4d(self, B, K, device):
        m = torch.zeros((B, K), dtype=torch.bool, device=device)
        m[:, self.idxs[0]: self.idxs[-1] + 1] = True
        return m.view(B, 1, 1, K)

    def k_hook(self):
        def hook(module, inputs, output):
            hs = inputs[0]
            k = output
            B, T, Dk = k.shape
            device = k.device
            if T <= 1:
                return output

            with torch.no_grad():
                q = self.attn.q_proj(hs)

            H = self.num_heads
            Hkv = self.num_kv_heads
            Dh = self.head_dim
            Dh_kv = self.kv_head_dim

            q = q.view(B, T, H, Dh).permute(0,2,1,3).contiguous()
            k_heads = k.view(B, T, Hkv, Dh_kv).permute(0,2,1,3).contiguous()

            if Hkv != H:
                rep = H // Hkv
                k_for_attn = k_heads.repeat_interleave(rep, dim=1)
            else:
                k_for_attn = k_heads

            scale = 1.0 / math.sqrt(Dh)
            logits = torch.matmul(q, k_for_attn.transpose(-2, -1)) * scale
            probs = F.softmax(logits, dim=-1)

            vmask4d = self._build_visual_mask_4d(B, K=T, device=device)
            H_vis = _visual_entropy(probs, vmask4d)
            H_head = H_vis.mean(dim=0)

            H_head_f = H_head.to(torch.float32)
            H_mean = H_head_f.mean()
            H_max = H_head_f.max()
            H_min = H_head_f.min()

            if self.direct == 1:
                tau_global = H_max / (H_mean + 1e-6) * self.d_a
            else:
                tau_global = H_min / (H_mean + 1e-6) * self.d_b

            scale_global = (1.0 / tau_global).item()
            vmask4d_kv = vmask4d.expand(B, Hkv, 1, T).transpose(-1, -2)
            scale_tensor = k_heads.new_full((1,1,1,1), scale_global)

            k_heads_scaled = torch.where(
                vmask4d_kv,
                k_heads * scale_tensor,
                k_heads
            )
            k_out = k_heads_scaled.permute(0,2,1,3).contiguous().view(B,T,Dk)
            return k_out
        return hook
