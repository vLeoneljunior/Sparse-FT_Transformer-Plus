from typing import List

import scipy.stats
import rtdl_lib as rtdl
import torch


class SaveAttentionMaps:
    def __init__(self):
        self.attention_maps: List[torch.Tensor] = []

    def __call__(self, _, __, output):
        self.attention_maps.append(output[1]['attention_probs'])


# Prepare data and model.
n_objects = 12
n_features = 2
X = torch.randn(n_objects, n_features)
model = rtdl.FTTransformer.make_default(
    n_num_features=n_features, cat_cardinalities=None, d_out=1
)

# The following hook will save all attention maps from all attention modules.
hook = SaveAttentionMaps()
for block in model.transformer.blocks:
    block['attention'].register_forward_hook(hook)

# Apply the model to all objects.
model.eval()
with torch.inference_mode():
    batch_size = 4
    for batch in X.split(batch_size):
        model(batch, None)

# Collect attention maps
n_blocks = len(model.transformer.blocks)
n_heads = model.transformer.blocks[0]['attention'].n_heads
n_tokens = n_features + 1
attention_maps = torch.cat(hook.attention_maps)
assert attention_maps.shape == (n_objects * n_blocks * n_heads, n_tokens, n_tokens)

# Calculate feature importance and ranks.
average_attention_map = attention_maps.mean(0)
average_cls_attention_map = average_attention_map[-1]  # consider only the [CLS] token
feature_importance = average_cls_attention_map[:-1]  # drop the [CLS] token importance
assert feature_importance.shape == (n_features,)

feature_ranks = scipy.stats.rankdata(-feature_importance.numpy())
feature_indices_sorted_by_importance = feature_importance.argsort(descending=True).numpy()
print(feature_ranks)
print(feature_indices_sorted_by_importance)