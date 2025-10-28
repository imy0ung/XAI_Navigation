# numpy
import numpy as np

# modeling
from vision_models.base_model import BaseModel
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import clip
from PIL import Image

# typing
from typing import List, Optional


class GradEclipSoftmaxModel(BaseModel):
    def __init__(self, model_name="ViT-B/16", device="cuda", temperature=1.0):
        super().__init__()
        self.device = device
        self.feature_dim = 1  # Grad-ECLIP heatmap은 1차원
        self.temperature = temperature  # Softmax temperature scaling
        
        # CLIP 모델 로드
        self.clip_model, self.preprocess = clip.load(model_name, device=device)
        # CLIP 모델을 float32로 유지 (half precision 문제 해결)
        self.clip_model = self.clip_model.float()
        self.clip_inres = self.clip_model.visual.input_resolution  # 이미지 해상도
        self.clip_ksize = self.clip_model.visual.conv1.kernel_size  # 패치 사이즈 (16 x 16)
        
        # 이미지 전처리
        self._transform = Compose([
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        
        print(f"[Grad-ECLIP-Softmax] CLIP resolution: {self.clip_inres}")
        print(f"[Grad-ECLIP-Softmax] CLIP kernel size: {self.clip_ksize}")
        print(f"[Grad-ECLIP-Softmax] Feature dimension: {self.feature_dim}")
        print(f"[Grad-ECLIP-Softmax] Softmax temperature: {self.temperature}")

    def imgprocess(self, img, patch_size=[16, 16], scale_factor=1):
        """이미지를 CLIP 모델에 맞게 전처리"""
        w, h = img.size
        ph, pw = patch_size
        nw = int(w * scale_factor / pw + 0.5) * pw
        nh = int(h * scale_factor / ph + 0.5) * ph

        ResizeOp = Resize((nh, nw), interpolation=InterpolationMode.BICUBIC)
        img = ResizeOp(img).convert("RGB")
        return self._transform(img)

    def attention_layer(self, q, k, v, num_heads=1):
        """Compute 'Scaled Dot Product Attention'"""
        tgt_len, bsz, embed_dim = q.shape
        head_dim = embed_dim // num_heads
        scaling = float(head_dim) ** -0.5
        q = q * scaling

        q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_heads = torch.bmm(attn_output_weights, v)
        assert list(attn_output_heads.size()) == [bsz * num_heads, tgt_len, head_dim]
        attn_output = attn_output_heads.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, -1)
        attn_output_weights = attn_output_weights.sum(dim=1) / num_heads
        return attn_output, attn_output_weights

    def clip_encode_dense(self, x, n):
        """CLIP 모델의 dense features 추출"""
        vision_width = self.clip_model.visual.transformer.width
        vision_heads = vision_width // 64

        # modified from CLIP
        # 입력을 float32로 유지
        if x.dtype != torch.float32:
            x = x.float()
        x = self.clip_model.visual.conv1(x)
        feah, feaw = x.shape[-2:]

        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        class_embedding = self.clip_model.visual.class_embedding.to(x.dtype)
        x = torch.cat([class_embedding + torch.zeros(x.shape[0], 1, x.shape[-1]).to(x), x], dim=1)

        # scale position embedding as the image w-h ratio
        pos_embedding = self.clip_model.visual.positional_embedding.to(x.dtype)
        tok_pos, img_pos = pos_embedding[:1, :], pos_embedding[1:, :]
        pos_h = self.clip_inres // self.clip_ksize[0]
        pos_w = self.clip_inres // self.clip_ksize[1]
        assert img_pos.size(0) == (pos_h * pos_w), f"the size of pos_embedding ({img_pos.size(0)}) does not match resolution shape pos_h ({pos_h}) * pos_w ({pos_w})"
        img_pos = img_pos.reshape(1, pos_h, pos_w, img_pos.shape[1]).permute(0, 3, 1, 2)
        img_pos = torch.nn.functional.interpolate(img_pos, size=(feah, feaw), mode='bicubic', align_corners=False)
        img_pos = img_pos.reshape(1, img_pos.shape[1], -1).permute(0, 2, 1)
        pos_embedding = torch.cat((tok_pos[None, ...], img_pos), dim=1)
        x = x + pos_embedding
        x = self.clip_model.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = torch.nn.Sequential(*self.clip_model.visual.transformer.resblocks[:-n])(x)

        attns = []
        atten_outs = []
        vs = []
        qs = []
        ks = []
        for TR in self.clip_model.visual.transformer.resblocks[-n:]:
            x_in = x
            x = TR.ln_1(x_in)
            linear = torch._C._nn.linear
            q, k, v = linear(x, TR.attn.in_proj_weight, TR.attn.in_proj_bias).chunk(3, dim=-1)
            attn_output, attn = self.attention_layer(q, k, v, 1)  # vision_heads=1
            attns.append(attn)
            atten_outs.append(attn_output)
            vs.append(v)
            qs.append(q)
            ks.append(k)

            x_after_attn = linear(attn_output, TR.attn.out_proj.weight, TR.attn.out_proj.bias)
            x = x_after_attn + x_in
            x = x + TR.mlp(TR.ln_2(x))

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip_model.visual.ln_post(x)
        x = x @ self.clip_model.visual.proj
        return x, x_in, vs, qs, ks, attns, atten_outs, (feah, feaw)

    def sim_qk(self, q, k):
        """Query-Key similarity 계산"""
        q_cls = F.normalize(q[:1, 0, :], dim=-1)
        k_patch = F.normalize(k[1:, 0, :], dim=-1)

        cosine_qk = (q_cls * k_patch).sum(-1)
        cosine_qk_max = cosine_qk.max(dim=-1, keepdim=True)[0]
        cosine_qk_min = cosine_qk.min(dim=-1, keepdim=True)[0]
        cosine_qk = (cosine_qk - cosine_qk_min) / (cosine_qk_max - cosine_qk_min)
        return cosine_qk

    def grad_eclip(self, c, qs, ks, vs, attn_outputs, map_size):
        """Grad-ECLIP heatmap 생성 + Softmax"""
        # gradient on last attention output
        tmp_maps = []
        for q, k, v, attn_output in zip(qs, ks, vs, attn_outputs):
            grad = torch.autograd.grad(
                c,
                attn_output,
                retain_graph=True)[0]

            grad_cls = grad[:1, 0, :]
            v_patch = v[1:, 0, :]
            cosine_qk = self.sim_qk(q, k).reshape(-1)
            tmp_maps.append((grad_cls * v_patch * cosine_qk[:, None]).sum(-1))

        emap = F.relu_(torch.stack(tmp_maps, dim=0)).sum(0)
        
        # 🔑 핵심: 개선된 Softmax 적용!
        # 1. Raw values 저장 (디버깅용)
        raw_min, raw_max = emap.min(), emap.max()
        
        # 2. Heatmap을 flatten하여 전체에 softmax 적용
        emap_flat = emap.flatten()
        emap_softmax = torch.softmax(emap_flat / self.temperature, dim=0)
        emap = emap_softmax.reshape(*map_size)
        
        # 3. 시각화 개선을 위한 스케일링 (선택적)
        # Softmax 값들을 [0, 1] 범위에서 더 잘 보이도록 조정
        if emap.max() > 0:
            # 상위 5% 값을 기준으로 스케일링
            top_5_percent = torch.quantile(emap.flatten(), 0.95)
            if top_5_percent > 0:
                emap_scaled = torch.clamp(emap / top_5_percent, 0, 1)
            else:
                emap_scaled = emap
        else:
            emap_scaled = emap
        
        # 4. 디버깅 정보 출력
        print(f"[Grad-ECLIP-Softmax] Raw range: [{raw_min:.3f}, {raw_max:.3f}]")
        print(f"[Grad-ECLIP-Softmax] Softmax range: [{emap.min():.6f}, {emap.max():.6f}], Sum: {emap.sum():.6f}")
        print(f"[Grad-ECLIP-Softmax] Scaled range: [{emap_scaled.min():.6f}, {emap_scaled.max():.6f}]")
        print(f"[Grad-ECLIP-Softmax] Temperature: {self.temperature}")
        
        # 5. 시각화용으로 스케일된 버전 반환
        return emap_scaled

    def get_image_features(self, image: np.ndarray, query_text: str = "toilet") -> torch.Tensor:
        """
        Grad-ECLIP heatmap 생성 (Softmax 적용)
        Args:
            image: (B, C, H, W) 또는 (C, H, W) 또는 (H, W, C) 형태의 이미지
            query_text: 쿼리 텍스트 (예: "toilet")
        Returns:
            heatmap: (1, H, W) 형태의 확률적 relevance map
        """
        # 원본 이미지 크기 저장 (resize용)
        original_shape = image.shape
        
        # 이미지 형태 정규화 - 완전히 새로운 배열 생성
        if len(image.shape) == 4:  # (B, C, H, W)
            img_array = np.array(image[0], copy=True, dtype=np.uint8)  # 첫 번째 배치만 사용하고 복사
            h, w = original_shape[2], original_shape[3]  # (B, C, H, W)에서 H, W 추출
        elif len(image.shape) == 3:  # (C, H, W) 또는 (H, W, C)
            if image.shape[0] == 3:  # (C, H, W)
                img_array = np.array(image.transpose(1, 2, 0), copy=True, dtype=np.uint8)  # (H, W, C)로 변환하고 복사
                h, w = original_shape[1], original_shape[2]  # (C, H, W)에서 H, W 추출
            else:  # (H, W, C) 형태는 그대로 복사
                img_array = np.array(image, copy=True, dtype=np.uint8)
                h, w = original_shape[0], original_shape[1]  # (H, W, C)에서 H, W 추출
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
        
        # numpy array를 PIL Image로 변환
        if img_array.dtype != np.uint8:
            # 정규화된 값 [0,1]을 [0,255]로 변환
            if img_array.max() <= 1.0:
                img_array = (img_array * 255).astype(np.uint8)
            else:
                img_array = img_array.astype(np.uint8)
        
        # PIL Image 생성
        img_pil = Image.fromarray(img_array)
        
        # 이미지 전처리
        img_preprocessed = self.imgprocess(img_pil).to(self.device).unsqueeze(0)
        
        # CLIP dense features 추출
        outputs, last_feat, vs, qs, ks, attns, atten_outs, map_size = self.clip_encode_dense(img_preprocessed, n=1)
        
        # 텍스트 임베딩 생성
        text_processed = clip.tokenize([query_text]).to(self.device)
        text_embedding = self.clip_model.encode_text(text_processed)
        text_embedding = F.normalize(text_embedding, dim=-1)
        
        # 이미지 임베딩과의 cosine similarity 계산
        img_embedding = F.normalize(outputs[:, 0], dim=-1)
        cosine = (img_embedding @ text_embedding.T)[0]
        
        # Grad-ECLIP heatmap 생성 (Softmax 적용!)
        heatmap = self.grad_eclip(cosine[0], qs, ks, vs, atten_outs, map_size)
        
        # 원본 이미지 크기로 resize (저장된 h, w 사용)
        resize = T.Resize((h, w))
        heatmap = resize(heatmap.unsqueeze(0)).squeeze(0)
        
        # (H, W) -> (1, H, W) 형태로 반환 (OneMap에서 사용할 형태)
        # heatmap은 이미 self.device에 있으므로 추가 이동 불필요
        heatmap_result = heatmap.unsqueeze(0)
        
        print(f"[Grad-ECLIP-Softmax] Final heatmap - Shape: {heatmap_result.shape}, Range: [{heatmap_result.min():.6f}, {heatmap_result.max():.6f}]")
        
        return heatmap_result

    def get_text_features(self, texts: List[str]) -> torch.Tensor:
        """
        텍스트 features 추출 (BaseModel 인터페이스 구현용)
        Grad-ECLIP에서는 실제로 사용하지 않음
        """
        text_processed = clip.tokenize(texts).to(self.device)
        text_embedding = self.clip_model.encode_text(text_processed)
        return F.normalize(text_embedding, dim=-1)

    def compute_similarity(self, image_feats: torch.Tensor, text_feats: torch.Tensor) -> torch.Tensor:
        """
        이미지와 텍스트 features 간의 유사도 계산 (BaseModel 인터페이스 구현용)
        Grad-ECLIP의 경우 이미지 features가 이미 heatmap이므로 그대로 반환
        """
        # Grad-ECLIP heatmap은 이미 relevance score이므로 그대로 반환
        if len(image_feats.shape) == 3:  # (1, H, W)
            return image_feats.squeeze(0)  # (H, W)
        else:
            return image_feats

    def eval(self):
        """평가 모드로 설정"""
        self.clip_model.eval()
        return self


if __name__ == "__main__":
    # 테스트 코드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GradEclipSoftmaxModel(model_name="ViT-B/16", device=device, temperature=1.0)
    
    # 테스트 이미지 로드
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # 이미지 features 추출 (Grad-ECLIP-Softmax heatmap)
    heatmap = model.get_image_features(test_image, "toilet")
    print(f"Softmax Heatmap shape: {heatmap.shape}")
    print(f"Softmax Heatmap range: {heatmap.min():.6f} ~ {heatmap.max():.6f}")
    print(f"Softmax Heatmap sum: {heatmap.sum():.6f}")
    
    # OneMap에서 사용할 수 있는 형태인지 확인
    print(f"OneMap feature_dim=1과 호환: {heatmap.shape[0] == 1}")