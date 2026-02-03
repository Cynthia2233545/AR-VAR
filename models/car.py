import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F

from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from models.var import VAR


class FP32_Layernorm(nn.LayerNorm):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        origin_dtype = inputs.dtype
        return F.layer_norm(inputs.float(), self.normalized_shape, self.weight.float(), self.bias.float(),
                            self.eps).to(origin_dtype)


class ControlConditionEmbedding(nn.Module):
    def __init__(
        self,
        conditioning_embedding_channels: int,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
    ):
        super().__init__()

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

        self.conv_out = nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


class CAR(VAR):
    def __init__(
        self, 
        vae_local: VQVAE, 
        num_classes=1000, 
        depth=16, 
        embed_dim=1024,
        num_heads=16, 
        mlp_ratio=4., 
        drop_rate=0., 
        attn_drop_rate=0.,
        drop_path_rate=0.,
        norm_eps=1e-6, 
        shared_aln=False, 
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 10 steps by default
        flash_if_available=True, 
        fused_if_available=True,
        repa_layer=8,  # 默认对齐第8层
        repa_dim=1024,  # 需与UNet特征处理后的维度一致
        repa_loss_type='mse',  # 可选: 'mse', 'cosine', 'l1'
        repa_weight=1.0,  # REPA损失权重
        repa_pool_type='mean',  # 可选: 'mean', 'cls', 'max'
        use_repa=True  # 是否使用REPA对齐
    ):
        super(CAR, self).__init__(
            vae_local, 
            num_classes, 
            depth, 
            embed_dim, 
            num_heads, 
            mlp_ratio,
            drop_rate, 
            attn_drop_rate, 
            drop_path_rate, 
            norm_eps, 
            shared_aln,
            cond_drop_rate, 
            attn_l2_norm, 
            patch_nums, 
            flash_if_available, 
            fused_if_available
        )

        conv_in_kernel = 3
        conv_in_padding = (conv_in_kernel - 1) // 2
        self.car_control_convs = ControlConditionEmbedding(conditioning_embedding_channels=self.C)
        self.car_var_conv = nn.Conv2d(self.C, self.C, kernel_size=conv_in_kernel, padding=conv_in_padding)
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.car_blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, 
                shared_aln=shared_aln,
                block_idx=block_idx, 
                embed_dim=self.C, 
                norm_layer=norm_layer, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio,
                drop=drop_rate, 
                attn_drop=attn_drop_rate, 
                drop_path=dpr[block_idx],
                last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, 
                fused_if_available=fused_if_available,
            )
            for block_idx in range(depth // 2)
        ])

        car_norm_layer = FP32_Layernorm
        car_skip_norm = []
        car_skip_linear = []
        for _ in range(depth // 2):
            car_skip_norm.append(car_norm_layer(2 * self.C, elementwise_affine=True, eps=1e-6))
            car_skip_linear.append(nn.Linear(2 * self.C, self.C))
        self.car_skip_norm = nn.ModuleList(car_skip_norm)
        self.car_skip_linear = nn.ModuleList(car_skip_linear)
        
        # REPA对齐配置（适配UNet特征）
        self.repa_layer = repa_layer
        self.repa_dim = repa_dim
        self.repa_loss_type = repa_loss_type
        self.repa_weight = repa_weight
        self.repa_pool_type = repa_pool_type
        self.use_repa = use_repa
        
        # 对齐投影头（将模型特征投影到UNet特征维度）
        self.repa_projection = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, repa_dim),
            nn.GELU(),
            nn.Linear(repa_dim, repa_dim)
        )
        
        # UNet特征处理头（将4维特征图转换为2维特征向量）
        self.unet_feature_processor = nn.Sequential(
            # 处理UNet输出的[B, 1, 256, 256]特征
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # [B, 64, 256, 256]
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((16, 16)),  # 压缩空间维度 [B, 64, 16, 16]
            nn.Flatten(),  # 展平为 [B, 64*16*16=16384]
            nn.Linear(16384, repa_dim),  # 映射到目标维度 [B, repa_dim]
            nn.LayerNorm(repa_dim)
        )
        
        # ===================== 最小改动1：新增残差融合分支（拼接-归一化-线性映射） =====================
        # 仅为对齐层（repa_layer）配置专属融合分支，不新增跨层组件
        self.anatomy_fusion_branch = nn.Sequential(
            FP32_Layernorm(2 * self.C, elementwise_affine=True, eps=1e-6),  # 拼接后归一化
            nn.Linear(2 * self.C, self.C),  # 线性映射回原特征维度
            nn.LayerNorm(self.C, eps=norm_eps)  # 输出归一化，保证数值稳定
        )
        # 可学习融合权重（平衡原始特征与解剖调制特征，初始值0.5）
        self.fusion_weight = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        
        # ===================== 最小改动2：新增解剖特征到序列的映射层（维度适配） =====================
        self.anatomy_to_seq = nn.Linear(self.repa_dim, self.C)
        
        # 根据选择的损失类型初始化损失函数
        if repa_loss_type == 'mse':
            self.repa_criterion = nn.MSELoss()
        elif repa_loss_type == 'l1':
            self.repa_criterion = nn.L1Loss()
        elif repa_loss_type == 'cosine':
            self.repa_criterion = lambda x, y: 1.0 - F.cosine_similarity(x, y, dim=1).mean()
        else:
            raise ValueError(f"Unsupported repa_loss_type: {repa_loss_type}")
    
    @torch.no_grad()
    def car_inference(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, control_tensors=None, unet_features=None
    ) -> torch.Tensor:
        # 保持与forward相同的UNet特征处理逻辑
        processed_unet = None
        if unet_features is not None and self.use_repa:
            processed_unet = self.unet_feature_processor(unet_features)
        
        # 其余代码保持不变...
        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng

        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B,
                                 device=self.lvl_1L.device)

        sos = cond_BD = self.class_emb(
            torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))

        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, : self.first_l]

        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])

        control_f = []
        if control_tensors is not None:
            assert control_tensors[0].shape[0] == B
            for control_tensor in control_tensors:
                control_i = self.car_control_convs(control_tensor)
                control_f.append(control_i)

        for cb in self.car_blocks:
            cb.attn.kv_caching(True)

        next_control_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1)

        for b in self.blocks:
            b.attn.kv_caching(True)

        intermediate_features = []
        
        for si, pn in enumerate(self.patch_nums):
            ratio = si / self.num_stages_minus_1
            cur_L += pn * pn
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map

            control_residual_f = []
            if control_tensors is not None:
                control_x = control_f[si].repeat(2, 1, 1, 1)
                var_x = next_control_token_map.transpose(1, 2).contiguous().reshape(2 * B, self.C, pn, pn)
                var_x = self.car_var_conv(var_x)
                control_x = var_x + control_x
                control_x = control_x.view(2 * B, self.C, -1).transpose(1, 2)
                control_x = control_x + lvl_pos[:, cur_L - pn * pn: cur_L]

                for cb in self.car_blocks:
                    control_x = cb(x=control_x, cond_BD=cond_BD_or_gss, attn_bias=None)
                    control_residual_f.append(control_x)

            for bidx, b in enumerate(self.blocks):
                if control_tensors is not None and bidx >= len(self.blocks) // 2:
                    con_f = control_residual_f.pop()
                    cat = torch.cat([x, con_f], dim=-1)
                    cat = self.car_skip_norm[bidx - len(self.blocks) // 2](cat)
                    x = self.car_skip_linear[bidx - len(self.blocks) // 2](cat)
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                
                # ===================== 最小改动3：推理阶段同步残差融合（与训练一致） =====================
                if self.use_repa and processed_unet is not None and bidx == self.repa_layer:
                    # 1. 解剖特征映射到序列维度
                    anatomy_seq = self.anatomy_to_seq(processed_unet)  # [B, C]
                    anatomy_seq = anatomy_seq.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, seq_len, C]
                    # 2. 拼接-归一化-线性映射
                    x_cat = torch.cat([x, anatomy_seq], dim=-1)  # [B, seq_len, 2C]
                    anatomy_modulate = self.anatomy_fusion_branch(x_cat)  # [B, seq_len, C]
                    # 3. 残差注入（动态权重平衡）
                    x = x * (1 - self.fusion_weight) + anatomy_modulate * self.fusion_weight
                
                if bidx == self.repa_layer:
                    intermediate_features.append(x.clone())

            logits_BlV = self.get_logits(x, cond_BD)

            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth:
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)
            else:
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ \
                         self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)

            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums),
                                                                                          f_hat, h_BChw)
            if si != self.num_stages_minus_1:
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_control_token_map = self.word_embed(next_token_map).repeat(2, 1, 1)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:,
                                                                   cur_L:cur_L + self.patch_nums[si + 1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)

        for b in self.blocks:
            b.attn.kv_caching(False)

        for cb in self.car_blocks:
            cb.attn.kv_caching(False)
        
        if unet_features is not None and self.use_repa and intermediate_features:
            if self.repa_pool_type == 'mean':
                pooled_feature = intermediate_features[-1].mean(1)
            elif self.repa_pool_type == 'cls':
                pooled_feature = intermediate_features[-1][:, 0]
            elif self.repa_pool_type == 'max':
                pooled_feature = intermediate_features[-1].max(1)[0]
            else:
                raise ValueError(f"Unsupported repa_pool_type: {self.repa_pool_type}")
            
            repa_feature = self.repa_projection(pooled_feature)

        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)

    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor,
                control_tensors=None, unet_features=None):
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        B = x_BLCv_wo_first_l[0].shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)

            control_f = []
            if control_tensors is not None:
                assert control_tensors[0].shape[0] == B
                for control_tensor in control_tensors:
                    control_tensor = control_tensor.to('cuda')
                    control_i = self.car_control_convs(control_tensor)
                    control_f.append(control_i)
            
            car_input = []
            var_x = sos.transpose(1, 2).contiguous().reshape(B, self.C, self.patch_nums[0], self.patch_nums[0])
            var_x = self.car_var_conv(var_x)
            car_x = var_x + control_f[0]
            car_x = car_x.view(B, self.C, -1).transpose(1, 2).contiguous()
            car_input.append(car_x)
            for si, (pn, var_input) in enumerate(zip(self.patch_nums[1:], x_BLCv_wo_first_l)):
                var_x = self.word_embed(var_input.float())
                var_x = var_x.transpose(1, 2).contiguous().reshape(B, self.C, pn, pn)
                var_x = self.car_var_conv(var_x)
                car_x = var_x + control_f[si+1]
                car_x = car_x.view(B, self.C, -1).transpose(1, 2).contiguous()
                car_input.append(car_x)

            car_input = torch.cat(car_input, dim=1)
            car_input += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]

            x_BLCv_wo_first_l = torch.cat(x_BLCv_wo_first_l, dim=1)

            x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed]

        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)

        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype

        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)

        control_residual_f = []
        for cb in self.car_blocks:
            car_input = cb(x=car_input, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            control_residual_f.append(car_input)

        intermediate_feature = None
        # ===================== 最小改动4：预处理UNet特征（用于后续融合） =====================
        processed_unet = None
        if self.use_repa and unet_features is not None:
            processed_unet = self.unet_feature_processor(unet_features)
        
        for i, b in enumerate(self.blocks):
            if i >= len(self.blocks) // 2:
                con_f = control_residual_f.pop()
                cat = torch.cat([x_BLC, con_f], dim=-1)
                cat = self.car_skip_norm[i - len(self.blocks) // 2](cat)
                x_BLC = self.car_skip_linear[i - len(self.blocks) // 2](cat)
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
            
            if i == self.repa_layer:
                intermediate_feature = x_BLC.clone()
                
                # ===================== 核心改动：对齐后执行残差注入式嵌入（拼接-归一化-线性映射） =====================
                if self.use_repa and processed_unet is not None:
                    # 步骤1：解剖特征维度适配（2D向量 → Transformer序列特征）
                    # [B, repa_dim] → [B, C] → [B, seq_len, C]（与当前层特征格式匹配）
                    anatomy_seq = self.anatomy_to_seq(processed_unet)  # [B, C]
                    anatomy_seq = anatomy_seq.unsqueeze(1).expand(-1, x_BLC.shape[1], -1)  # [B, seq_len, C]
                    
                    # 步骤2：拼接原始特征与解剖特征（保留完整信息）
                    x_blc_cat = torch.cat([x_BLC, anatomy_seq], dim=-1)  # [B, seq_len, 2C]
                    
                    # 步骤3：归一化+线性映射（残差分支处理，保证维度匹配）
                    anatomy_modulate = self.anatomy_fusion_branch(x_blc_cat)  # [B, seq_len, C]
                    
                    # 步骤4：残差注入（动态权重平衡，不淹没原始生成特征）
                    x_BLC = x_BLC * (1 - self.fusion_weight) + anatomy_modulate * self.fusion_weight

        logits_BLV = self.get_logits(x_BLC.float(), cond_BD)
        
        # 计算REPA损失（处理UNet特征维度）
        repa_loss = None
        if self.use_repa and intermediate_feature is not None and processed_unet is not None:
            # 1. 处理模型中间特征
            if self.repa_pool_type == 'mean':
                pooled_feature = intermediate_feature.mean(1)
            elif self.repa_pool_type == 'cls':
                pooled_feature = intermediate_feature[:, 0]
            elif self.repa_pool_type == 'max':
                pooled_feature = intermediate_feature.max(1)[0]
            else:
                raise ValueError(f"Unsupported repa_pool_type: {self.repa_pool_type}")
            
            # 2. 投影到目标维度并计算损失
            projected_feature = self.repa_projection(pooled_feature)
            repa_loss = self.repa_criterion(projected_feature, processed_unet) * self.repa_weight

        return logits_BLV, repa_loss