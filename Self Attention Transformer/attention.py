import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===================================
# 1. Простая Scaled Dot-Product Attention
# ===================================
def Attn(Q, K, V):                                        # Q: (B,N,E); K,V: (B,M,E)
    E_dim = Q.size(-1)                                 # размерность эмбединга E
    W = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(E_dim)  # (B,N,M)      
    W = F.softmax(W, dim = -1)                          # по последнему индексу
    return torch.bmm(W, V)                             # (B,N,E)


# ===================================
# 2. Полная MultiHeadAttention 
# ===================================
def MultiHeadAttention(Q, K, V, 
                     Wq, Wk, Wv, Wo,
                     Bq=None, Bk=None, Bv=None, Bo=None,
                     key_mask=None, attn_mask=None,
                     num_heads=8, embed_dim=512):
    """
    Полная ручная реализация, идентичная nn.MultiheadAttention
    Q: (N, B, E)   — seq_len, batch, embed_dim
    """
    N, B, E = Q.shape
    M = K.shape[0]
    H = num_heads
    Eh = embed_dim // H
    
    # Линейные проекции
    q = F.linear(Q, Wq, Bq)      # (N,B,E)
    k = F.linear(K, Wk, Bk)      # (M,B,E)
    v = F.linear(V, Wv, Bv)      # (M,B,E)
    
    # Разбиение на головы 
    q = q.view(N, B * H, Eh).transpose(0, 1)   # (B*H, N, Eh)
    k = k.view(M, B * H, Eh).transpose(0, 1)   # (B*H, M, Eh)
    v = v.view(M, B * H, Eh).transpose(0, 1)   # (B*H, M, Eh)
    
    # Scaled dot-product
    W = torch.bmm(q, k.transpose(1, 2)) * (Eh ** -0.5)   # (B*H, N, M)
    
    # Маски
    if attn_mask is not None:
        W = W + attn_mask.unsqueeze(0)                # broadcasting по батчу+головам
    
    if key_mask is not None:
        W = W.view(B, H, N, M)
        key_mask = key_mask.unsqueeze(1).unsqueeze(2)  # (B,1,1,M)
        W = W.masked_fill(key_mask, float('-inf'))
        W = W.view(B * H, N, M)
    
    W = F.softmax(W, dim=-1)                              # (B*H, N, M)
    
    A = torch.bmm(W, v)                                 # (B*H, N, Eh)
    
    # Собираю головы обратно
    A = A.transpose(0, 1).contiguous().view(N, B, E)     # (N, B, E)
    attn_weights = W.view(B, H, N, M).mean(dim=1)         # усреднённые веса (B, N, M)
    
    return F.linear(A, Wo, Bo), attn_weights


# ===================================
# 3. Пример использования 
# ===================================
if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Параметры
    E, H, N, M, B = 100, 10, 3, 3, 1
    Ek, Ev = 100, 100
    
    Q = torch.rand(N, B, E)
    K = torch.rand(M, B, Ek)
    V = torch.rand(M, B, Ev)
    
    # Оригинальный PyTorch модуль
    MHA = nn.MultiheadAttention(E, H, kdim=Ek, vdim=Ev, batch_first=False)
    A1, W1 = MHA(Q, K, V)
    
    # Ручная реализация с теми же весами
    A2, W2 = MultiHeadAttention(
        Q, K, V,
        Wq = MHA.in_proj_weight[0:E],
        Wk = MHA.in_proj_weight[E:2*E],
        Wv = MHA.in_proj_weight[2*E:],
        
        Wo = MHA.out_proj.weight,
        Bq = MHA.in_proj_bias[0:E],
        Bk = MHA.in_proj_bias[E:2*E],
        Bv = MHA.in_proj_bias[2*E:],
        Bo = MHA.out_proj.bias,
        num_heads=H,
        embed_dim=E
    )
    
    print("Разница в выходе A:", ((A1 - A2)**2).sum().sqrt().item())   # ~1e-6
    print("Разница в весах W:", ((W1 - W2)**2).sum().sum().sqrt().item())
    
    # Пример с масками из статьи
    print("\nПример с масками:")
    E, H, N, M, B = 8, 4, 3, 4, 1
    Q = torch.rand(N, B, E)
    K = torch.rand(M, B, E)
    V = torch.rand(M, B, E)
    
    MHA2 = nn.MultiheadAttention(E, H, batch_first=False)
    
    key_padding_mask = torch.tensor([[False, True, False, True]])  # отключаем 2-й и 4-й ключ
    inf = float("-inf")
    attn_mask = torch.tensor([[inf, 0.0, 0.0, inf],
                          [0.0, 0.0, 0.0, 0.0],
                          [inf, 0.0, 0.0, inf]])
    
    A3, W3 = MHA2(Q, K, V, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
    print("Веса с масками (оригинал):")
    print(W3.round(decimals=4))
