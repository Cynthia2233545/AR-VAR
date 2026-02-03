import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.nn.functional as F

import dist
from models import CAR, VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor


class CARTrainer(object):
    def __init__(
        self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
        vae_local: VQVAE, car_wo_ddp: CAR, car: DDP,
        car_opt: AmpOptimizer, label_smooth: float,
        lambda_rep=0.5,  # REPA损失权重
        repa_loss_type='mse',  # REPA损失类型: 'mse', 'l1', 'cosine'
        use_repa=True  # 是否使用REPA损失
    ):
        super(CARTrainer, self).__init__()
        
        self.car, self.vae_local, self.quantize_local = car, vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.car_wo_ddp: CAR = car_wo_ddp  # after torch.compile
        self.car_opt = car_opt
        
        del self.car_wo_ddp.rng
        self.car_wo_ddp.rng = torch.Generator(device=device)
        
        self.label_smooth = label_smooth
        self.train_loss = nn.CrossEntropyLoss(label_smoothing=label_smooth, reduction='none')
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.lambda_rep = lambda_rep  # REPA损失权重
        self.repa_loss_type = repa_loss_type
        self.use_repa = use_repa
        
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
    def eval_ep(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        training = self.car_wo_ddp.training
        self.car_wo_ddp.eval()
        
        # 修改：在数据加载时解包unet_feature
        for inp_B3HW, control_tensors, label_B, unet_feature in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            unet_feature = unet_feature.to(dist.get_device(), non_blocking=True)  # 移动UNet特征到设备
            for idx in range(len(control_tensors)):
                control_tensors[idx] = control_tensors[idx].to(dist.get_device(), non_blocking=True)

            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

            # 修改：添加unet_features参数
            logits_BLV, repa_loss = self.car_wo_ddp(label_B, x_BLCv_wo_first_l, control_tensors, unet_features=unet_feature)

            L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            
            # 计算REPA损失
            if self.use_repa and repa_loss is not None:
                repa_loss_val = repa_loss.item() * B
                L_mean += repa_loss_val * self.lambda_rep
                L_tail += repa_loss_val * self.lambda_rep
            
            tot += B
        self.car_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt
    
    def train_step(
        self, it: int, stepping: bool, metric_lg: MetricLogger,
        inp_B3HW: FTen, control_tensors: List[FTen], 
        label_B: Union[ITen, FTen], unet_feature: FTen  # 改为UNet特征参数
    ) -> Tuple[Optional[Union[Ten, float]], Optional[float]]:
        # 确保UNet特征在正确设备上
        unet_feature = unet_feature.to(dist.get_device())
        
        # forward
        B, V = label_B.shape[0], self.vae_local.vocab_size
        self.car.require_backward_grad_sync = stepping

        gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)

        with self.car_opt.amp_ctx:
            # 获取logits和REPA损失，传入UNet特征
            logits_BLV, repa_loss = self.car(
                label_B, x_BLCv_wo_first_l, 
                control_tensors, 
                unet_features=unet_feature if self.use_repa else None  # 改为UNet特征参数
            )

            # 计算原始生成损失
            loss = self.train_loss(logits_BLV.view(-1, V), gt_BL.view(-1)).view(B, -1)
            lw = self.loss_weight
            original_loss = loss.mul(lw).sum(dim=-1).mean()
            
            # 总损失 = 原始损失 + REPA损失
            total_loss = original_loss
            repa_loss_val = 0.0
            if self.use_repa and repa_loss is not None:
                repa_loss_val = repa_loss.item()
                total_loss = total_loss + self.lambda_rep * repa_loss

        # backward
        grad_norm, scale_log2 = self.car_opt.backward_clip_step(loss=total_loss, stepping=stepping)
        
        # log
        pred_BL = logits_BLV.data.argmax(dim=-1)
        if it == 0 or it in metric_lg.log_iters:
            Lmean = self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)).item()
            acc_mean = (pred_BL == gt_BL).float().mean().item() * 100
            Ltail = self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)).item()
            acc_tail = (pred_BL[:, -self.last_l:] == gt_BL[:, -self.last_l:]).float().mean().item() * 100
            grad_norm = grad_norm.item()
            
            # 记录REPA损失
            metric_lg.update(
                Lm=Lmean, Lt=Ltail, Accm=acc_mean, Acct=acc_tail, 
                tnm=grad_norm, repa=repa_loss_val,
                orig_loss=original_loss.item(),
                total_loss=total_loss.item()
            )

        return grad_norm, scale_log2
    
    def get_config(self):
        return {
            'patch_nums': self.patch_nums,
            'resos': self.resos,
            'label_smooth': self.label_smooth,
            'lambda_rep': self.lambda_rep,  # 添加REPA配置
            'repa_loss_type': self.repa_loss_type,
            'use_repa': self.use_repa
        }
    
    def state_dict(self):
        state = {'config': self.get_config()}
        for k in ('car_wo_ddp', 'vae_local', 'car_opt'):
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                state[k] = m.state_dict()
        return state
    
    def load_state_dict(self, state, strict=True, skip_vae=False):
        for k in ('car_wo_ddp', 'vae_local', 'car_opt'):
            if skip_vae and 'vae' in k: continue
            m = getattr(self, k)
            if m is not None:
                if hasattr(m, '_orig_mod'):
                    m = m._orig_mod
                ret = m.load_state_dict(state[k], strict=strict)
                if ret is not None:
                    missing, unexpected = ret
                    print(f'[CARTrainer.load_state_dict] {k} missing:  {missing}')
                    print(f'[CARTrainer.load_state_dict] {k} unexpected:  {unexpected}')

        config: dict = state.pop('config', None)
        if config is not None:
            for k, v in self.get_config().items():
                if config.get(k, None) != v:
                    err = f'[CAR.load_state_dict] config mismatch:  this.{k}={v} (ckpt.{k}={config.get(k, None)})'
                    if strict: raise AttributeError(err)
                    else: print(err)
    