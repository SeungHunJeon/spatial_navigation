from fla.layers import DeltaNet
import torch
import torch.nn as nn
from platformdirs import user_cache_dir
from fla.models.utils import Cache


# class DeltaNetRNN(nn.Module):
#     def __init__(self, deltanet: DeltaNet):
#         super().__init__()
#         self.dn = deltanet
#
#     def forward(self, x, hidden=None):
#         # hidden = recurrent_state dict 그대로 전달
#         past = None
#         if hidden is not None:
#             past = { self.dn.layer_idx: hidden }
#
#         out, _, past_key_values = self.dn(
#             x,
#             past_key_values=past,
#             use_cache=True
#         )
#
#         # 다음 step에서 사용할 hidden 반환
#         next_hidden = past_key_values[self.dn.layer_idx]
#         return out, next_hidden

from fla.models.utils import Cache

if __name__ == "__main__":
    bs, num_heads, seq_len, hidden_size = 16, 1, 128, 64
    device, dtype = "cuda", torch.bfloat16

    gated_deltanet = (
        DeltaNet(
            hidden_size=hidden_size,
            num_heads=num_heads,
            mode='chunk',
            layer_idx=0,          # 👈 이거 추가
        )
        .to(device=device, dtype=dtype)
    )

    gated_deltanet.eval()

    x = torch.randn(bs, 1, hidden_size, device=device, dtype=dtype)
    x_batch = torch.stack([x] * seq_len, dim=1).squeeze(2)

    # Cache 초기화
    hidden_batch = Cache()

    out, _, hidden_batch = gated_deltanet(
        x_batch,
        past_key_values=hidden_batch,
        use_cache=True
    )

    hidden_recurrent = Cache()

    for t in range(seq_len):
        out_recurrent, _, hidden_recurrent = gated_deltanet(
            x,
            past_key_values=hidden_recurrent,
            use_cache=True
        )

        # recurrent_state 꺼내보기
        state = hidden_recurrent[gated_deltanet.layer_idx]['recurrent_state']
        print(
            f"Step {t}: out shape={out_recurrent.shape}, "
            f"recurrent_state shape={state.shape}"
        )
    print(torch.norm(out[:, -1, :] - out_recurrent[:, 0, :]))
