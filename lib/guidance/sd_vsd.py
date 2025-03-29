import cv2
import random
import numpy as np
from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

from .perpneg_utils import weighted_perpendicular_aggregator


class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        return torch.zeros([1], device=input_tensor.device, dtype=input_tensor.dtype)  # dummy loss value

    @staticmethod
    def backward(ctx, grad):
        gt_grad, = ctx.saved_tensors
        batch_size = len(gt_grad)
        return gt_grad / batch_size, None


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class StableDiffusion(nn.Module):
    def __init__(self, device, fp16=False, vram_O=False, sd_version='2.1', hf_key=None, t_range=[0.02, 0.98],
                 weighting_strategy='fantasia3d', opt=None, text=None):
        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.precision_t = torch.float16 if fp16 else torch.float32
        self.weighting_strategy = weighting_strategy
        self.opt = opt  # opt参数，与trainer.py中的opt参数一致
        self.text = text

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif self.sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif self.sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        # Create model
        self.vae = AutoencoderKL.from_pretrained(model_key, subfolder="vae").to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer")
        self.text_encoder = CLIPTextModel.from_pretrained(model_key, subfolder="text_encoder").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_key, subfolder="unet").to(self.device)

        # 添加对 xformers 的支持
        if is_xformers_available():
            self.unet.enable_xformers_memory_efficient_attention()

        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for convenience

        # 添加buffer相关变量
        self.buffer_imgs = None
        self.buffer_poses = None
        self.buffer_index = 0

        if self.opt.use_vsd:
            from .lora_unet import UNet2DConditionModel_custom     
            from diffusers.loaders import AttnProcsLayers
            from diffusers.models.attention_processor import LoRAAttnProcessor
            import einops
            if not self.opt.v_pred:
                _unet = UNet2DConditionModel_custom.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="unet", low_cpu_mem_usage=False, device_map=None).to(device)
            else:
                _unet = UNet2DConditionModel_custom.from_pretrained("stabilityai/stable-diffusion-2-1", subfolder="unet", low_cpu_mem_usage=False, device_map=None).to(device)
            
            _unet.requires_grad_(False)
            lora_attn_procs = {}
            for name in self.unet.attn_processors.keys():
                cross_attention_dim = None if name.endswith("attn1.processor") else _unet.config.cross_attention_dim
                if name.startswith("mid_block"):
                    hidden_size = _unet.config.block_out_channels[-1]
                elif name.startswith("up_blocks"):
                    block_id = int(name[len("up_blocks.")])
                    hidden_size = list(reversed(_unet.config.block_out_channels))[block_id]
                elif name.startswith("down_blocks"):
                    block_id = int(name[len("down_blocks.")])
                    hidden_size = _unet.config.block_out_channels[block_id]
                lora_attn_procs[name] = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim)
            _unet.set_attn_processor(lora_attn_procs)
            lora_layers = AttnProcsLayers(_unet.attn_processors)

            text_input = self.tokenizer(self.text, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')
            with torch.no_grad():
                text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
            
            class LoraUnet(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.unet = _unet
                    self.sample_size = _unet.sample_size
                    self.in_channels = _unet.in_channels
                    self.device = device
                    self.dtype = torch.float32
                    self.text_embeddings = text_embeddings
                def forward(self,x,t,c=None,shading="albedo"):
                    textemb = einops.repeat(self.text_embeddings, '1 L D -> B L D', B=x.shape[0]).to(device)
                    return self.unet(x,t,encoder_hidden_states=textemb,c=c,shading=shading)
            self._unet = _unet
            self.lora_layers = lora_layers
            self.q_unet = LoraUnet().to(device)

            # 设置训练参数
            # 获取所有LoRA参数，以及其他需要训练的参数
            # 获取LoRA参数（排除特定模块的参数）
            lora_params = []
            for name, param in self._unet.named_parameters():
                if 'camera_emb' not in name and 'lambertian_emb' not in name and 'textureless_emb' not in name and 'normal_emb' not in name:
                    lora_params.append(param)
                    
            other_params = [
                {'params': self._unet.camera_emb.parameters()},
                {'params': self._unet.lambertian_emb},
                {'params': self._unet.textureless_emb},
                {'params': self._unet.normal_emb},
            ]
            
            params = list(other_params)
            params.append({'params': lora_params})
            
            self.q_unet_optimizer = torch.optim.AdamW(params, lr=self.opt.unet_lr)  # naive adam
            warm_up_lr_unet = lambda iter: iter / (self.opt.warm_iters*self.opt.K+1) if iter <= (self.opt.warm_iters*self.opt.K+1) else 1
            self.q_unet_scheduler = torch.optim.lr_scheduler.LambdaLR(self.q_unet_optimizer, warm_up_lr_unet)

        '''
            初始化q_unet完成
        '''
        self.latents = None

        print(f'[INFO] loaded stable diffusion!')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        """
        Args:
            prompt: str

        Returns:
            text_embeddings: torch.Tensor
        """
        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt,
                                    padding='max_length',
                                    max_length=self.tokenizer.model_max_length,
                                    truncation=True,
                                    return_tensors='pt')
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        return text_embeddings

    # def get_text_embeds(self, prompt, negative_prompt):
    #     # prompt, negative_prompt: [str]

    #     # Tokenize text and get embeddings
    #     text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt')

    #     with torch.no_grad():
    #         text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

    #     # Do the same for unconditional embeddings
    #     uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt')

    #     with torch.no_grad():
    #         uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

    #     # Cat for final embeddings
    #     text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    #     return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100, t5=False, pose = None, shading = None, grad_clip = None, as_latent = False):
        
        # interp to 512x512 to be fed into vae.
        assert torch.isnan(pred_rgb).sum() == 0, print(pred_rgb)
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)        

        if t5: # Anneal time schedule
            t = torch.randint(self.min_step, 500 + 1, (latents.shape[0],), dtype=torch.long, device=self.device)
        else:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)
            print(f"text_embeddings shape: {text_embeddings.shape}")
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            '''
                利用q_unet预测噪声
            '''
            timesteps = torch.randint(0, 1000, (self.opt.unet_bs,), device=self.device).long()
            if self.opt.use_vsd:
                if self.q_unet is None:
                    raise NotImplementedError()
                if pose is None:
                    raise NotImplementedError()
                noise_pred_q = self.q_unet(latents_noisy, timesteps, c = pose, shading = shading).sample

                '''
                    采用v-prediction策略，替代传统的epsilon-prediction策略
                '''
                
                if self.opt.v_pred:
                    sqrt_alpha_prod = self.scheduler.alphas_cumprod.to(self.device)[t] ** 0.5
                    sqrt_alpha_prod = sqrt_alpha_prod.flatten()
                    while len(sqrt_alpha_prod.shape) < len(latents_noisy.shape):
                        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
                    sqrt_one_minus_alpha_prod = (1 - self.scheduler.alphas_cumprod.to(self.device)[t]) ** 0.5
                    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
                    while len(sqrt_one_minus_alpha_prod.shape) < len(latents_noisy.shape):
                        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)
                    noise_pred_q = sqrt_alpha_prod * noise_pred_q + sqrt_one_minus_alpha_prod * latents_noisy

        if self.weighting_strategy == "sds":
            # w(t), sigma_t^2
            w = (1 - self.alphas[t]).view(-1, 1, 1, 1)
        elif self.weighting_strategy == "fantasia3d":
            w = (self.alphas[t] ** 0.5 * (1 - self.alphas[t])).view(-1, 1, 1, 1)
        else:
            raise ValueError(
                f"Unknown weighting strategy: {self.cfg.weighting_strategy}"
            )
        
        if self.opt.use_vsd:
            grad = w * (noise_pred - noise_pred_q)
        else:
            grad = w * (noise_pred - noise)

        # 这里默认是不开启的，梯度裁剪，考虑可能稳定训练过程？
        if grad_clip is not None:
            grad = grad.clamp(-grad_clip, grad_clip)
        
        grad = torch.nan_to_num(grad)

        # 这里改为自定义梯度计算方法，BP计算梯度相同，但省去前向传播的计算，提高效率
        loss = SpecifyGradient.apply(latents, grad)

        #　pseudo_loss = torch.mul((w*noise_pred).detach(), latents.detach()).detach().sum()
        # return loss, pseudo_loss, latents
        
        self.latents = latents
        
        return loss
    
    def get_latents(self):
        return self.latents
    
    def produce_latents(self, text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if latents is None:
            latents = torch.randn((text_embeddings.shape[0] // 2, self.unet.in_channels, height // 8, width // 8), device=self.device)

        self.scheduler.set_timesteps(num_inference_steps)

        with torch.autocast('cuda'):
            for i, t in enumerate(self.scheduler.timesteps):
                # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
                latent_model_input = torch.cat([latents] * 2)

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

                # perform guidance
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents)['prev_sample']
        
        return latents

    def decode_latents(self, latents):

        latents = 1 / 0.18215 * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        
        return imgs

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents
    
    def train_q_unet(self, global_step, data, pred_rgb, shading='albedo', as_latent=False):
        """
            更新q_unet网络参数
        """
        assert torch.isnan(pred_rgb).sum() == 0, print(pred_rgb)
        if as_latent:
            latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)
        else:
            pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.encode_imgs(pred_rgb_512)        
        
        for _ in range(self.opt.K):
            self.q_unet_optimizer.zero_grad()
            # 这里还没有使用更高级的时间步策略
            timesteps = torch.randint(0, 1000, (self.opt.unet_bs,), device=self.device).long()
            
            with torch.no_grad():
                latents_clean = latents.expand(self.opt.unet_bs, latents.shape[1], latents.shape[2], latents.shape[3]).contiguous()
                pose = data['pose']
                pose = pose.expand(self.opt.unet_bs, 16).contiguous()
                if random.random() < self.opt.uncond_p:
                    pose = torch.zeros_like(pose)
            
            noise = torch.randn(latents_clean.shape, device=self.device)
            latents_noisy = self.scheduler.add_noise(latents_clean, noise, timesteps)
            
            model_output = self.q_unet(latents_noisy, timesteps, c=pose, shading=shading).sample

            loss_q_unet = F.mse_loss(model_output, self.scheduler.get_velocity(latents_clean, noise, timesteps))
            
            return loss_q_unet
                
    def update_q_unet(self, loss_q_unet):
                
        loss_q_unet.backward()
        self.q_unet_optimizer.step()
        
        if self.opt.q_scheduler_update_every_step:
            self.q_unet_scheduler.step()
            


    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
