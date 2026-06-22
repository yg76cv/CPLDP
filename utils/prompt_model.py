import torch
import torch.nn as nn
import clip

class ParameterWrapper(nn.Module):
    def __init__(self, parameter):
        super(ParameterWrapper, self).__init__()
        self.parameter = nn.Parameter(parameter.data)

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class Prompts(nn.Module):
    def __init__(self, clip_model, length_prompt=16):
        super().__init__()
        if not 1 <= length_prompt <= 75:
            raise ValueError("length_prompt must be between 1 and 75")
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_template = " ".join(["X"] * length_prompt)
        self.embedding_prompts = nn.ModuleList()
        clip_dev = next(clip_model.parameters()).device
        tokenized = clip.tokenize([self.prompt_template]).to(clip_dev)

        with torch.no_grad():
            emb_init = clip_model.token_embedding(tokenized).to(clip_dev)
        for _ in range(5):
            emb = emb_init + 0.01 * torch.randn_like(emb_init)
            self.embedding_prompts.append(ParameterWrapper(emb))

        with torch.no_grad():
            ref_emb_init = clip_model.token_embedding(tokenized).to(clip_dev)

        self.register_buffer("ref_embedding_init", ref_emb_init.clone())
        self.ref_embedding = nn.Parameter(ref_emb_init + 0.01 * torch.randn_like(ref_emb_init))

    def tokenize_templates(self, count, device):
        return clip.tokenize([self.prompt_template] * count).to(device)

    def forward(self, inputs):
        tensor = inputs["tensor"]
        flag = inputs.get("flag", 1)

        tokenized_prompts = self.tokenize_templates(5, tensor.device)
        prompt_stack = torch.cat([p.parameter for p in self.embedding_prompts], dim=0)
        text_features = self.text_encoder(prompt_stack, tokenized_prompts)

        tokenized_ref = self.tokenize_templates(1, tensor.device)
        ref_feat = self.text_encoder(self.ref_embedding, tokenized_ref)

        image_features = nn.functional.normalize(tensor, dim=-1)
        text_features = nn.functional.normalize(text_features, dim=-1)
        ref_feat = nn.functional.normalize(ref_feat, dim=-1)

        similarity = 100.0 * (image_features @ text_features.T)
        ref_sim = 100.0 * (image_features @ ref_feat.T)
        logits_6 = torch.cat([similarity, ref_sim], dim=1)

        if flag == 0:
            return {"degradation": logits_6, "reference": ref_feat}
        return {"degradation": nn.functional.softmax(logits_6, dim=-1), "reference": ref_feat}
