import os
import torch
import torch.nn as nn
import transformers
from agents.peract_bimanual.trajectory_gpt2 import GPT2Model
import torch.nn.functional as F

def r2bc_debug_enabled() -> bool:
    return os.environ.get("R2BC_VERBOSE_DEBUG", "0") == "1"


def r2bc_debug_print(*args, **kwargs):
    if r2bc_debug_enabled():
        print(*args, **kwargs)

def _scale_stat(name, x):
    if not torch.is_tensor(x):
        r2bc_debug_print(f"[scale debug] {name}: non-tensor {type(x)}")
        return
    xd = x.detach().float()
    r2bc_debug_print(
        f"[scale debug] {name}: "
        f"shape={tuple(xd.shape)} "
        f"mean={xd.mean().item():+.4e} "
        f"std={xd.std().item():+.4e} "
        f"min={xd.min().item():+.4e} "
        f"max={xd.max().item():+.4e}"
    )

class SkillManager(nn.Module):
    def __init__(
            self,
            num_classes,
            embedding_matrix=None,
            voxel_dim=128,
            lang_dim=128,
            hidden_size=256,
            output_dim=18,
            max_voxels=8000,
            max_lang_tokens=77,
            **kwargs):
        super().__init__()

        self.hidden_size = hidden_size
        self.output_dim = output_dim

        # GPT-2 configuration
        config = transformers.GPT2Config(
            vocab_size=1,  # not used
            n_embd=hidden_size,
            n_head=4, 
            n_ctx=1077,
        )

        self.max_voxels = max_voxels
        self.max_lang_tokens = max_lang_tokens
        self.embed_voxel = nn.Linear(voxel_dim, hidden_size)
        self.embed_lang = nn.Linear(lang_dim, hidden_size)
        self.transformer = GPT2Model(config)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_logits = nn.Linear(hidden_size, output_dim)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_class = num_classes
        if embedding_matrix is not None:
            self.embeddings_matrix = embedding_matrix.to(self.device)

    def forward(self, voxel_embedding, language_embedding):
        debug_scale = os.environ.get("R2BC_SCALE_DEBUG", "0") == "1"

        if debug_scale:
            _scale_stat("skill_manager/input_voxel_embedding", voxel_embedding)
            _scale_stat("skill_manager/input_language_embedding", language_embedding)

        batch_size = voxel_embedding.shape[0]
        voxel_embeddings = self.embed_voxel(voxel_embedding)  # [b, 8000, hidden_size]
        language_embeddings = self.embed_lang(language_embedding)  # [b, 77, hidden_size]

        if debug_scale:
            _scale_stat("skill_manager/embed_voxel_before_pool", voxel_embeddings)
            _scale_stat("skill_manager/embed_lang", language_embeddings)

        voxel_embeddings = voxel_embeddings.permute(0, 2, 1)  # [b, hidden_size, 8000]
        voxel_embeddings = F.avg_pool1d(voxel_embeddings, kernel_size=16, stride=16)  # [b, hidden_size, 500]
        voxel_embeddings = voxel_embeddings.permute(0, 2, 1)  # [b, 500, hidden_size]
        inputs = torch.cat([language_embeddings, voxel_embeddings], dim=1)

        if debug_scale:
            _scale_stat("skill_manager/inputs_before_ln", inputs)

        stacked_inputs = self.embed_ln(inputs)

        if debug_scale:
            _scale_stat("skill_manager/inputs_after_ln", stacked_inputs)
        attention_mask = torch.ones(
            (batch_size, self.max_lang_tokens + self.max_voxels),
            device=voxel_embedding.device,
            dtype=torch.long  # Ensure correct dtype
        )
        assert torch.isfinite(attention_mask).all(), "attention_mask contains NaN or Inf"
        assert torch.all((attention_mask == 1)), "attention_mask contains values not equal to 1"
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=None,
        )

        hidden_state = transformer_outputs.last_hidden_state  # [b, 8077, hidden_size]
        aggregated_hidden = hidden_state.mean(dim=1)  # [b, hidden_size]
        logits = self.predict_logits(aggregated_hidden)  # [b, output_dim]

        temperature = float(os.environ.get("R2BC_SKILL_TEMP", "10.0"))
        probs = F.softmax(logits / temperature, dim=1)

        if debug_scale:
            _scale_stat("skill_manager/logits", logits)
            _scale_stat("skill_manager/probs", probs)

            r2bc_debug_print("[skill debug] skill_temperature:", temperature)
            r2bc_debug_print("[skill debug] logits:", logits.detach().cpu().tolist())

            topv, topi = torch.topk(probs.detach(), k=5, dim=1)
            entropy = -(probs.detach() * torch.log(probs.detach() + 1e-8)).sum(dim=1)

            r2bc_debug_print("[skill debug] probs_topi:", topi.cpu().tolist())
            r2bc_debug_print("[skill debug] probs_topv:", topv.cpu().tolist())
            r2bc_debug_print("[skill debug] probs_entropy:", entropy.cpu().tolist())

        skill = torch.matmul(probs, self.embeddings_matrix.to(probs.device))
        skill = skill.view(-1,77,512)
        return skill