"""
Basic OSTrack model.
"""
import math
import os
from typing import List
import cv2
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from mmcv.cnn import ConvModule
from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.denoising_diffusion_pytorch.simple_diffusion import *
from lib.models.language_model import build_bert
from  torchvision.utils import save_image
from .positional_encoding.untied.absolute import Untied2DPositionalEncoder ,UntiedPositionalEncoder
# from .positional_encoding.untied.relative import RelativePosition2DEncoder, generate_2d_concatenated_cross_attention_relative_positional_encoding_index,generate_2d_concatenated_cross_attention_relative_positional_encoding_index_p
from .cross_attention import CrossAttentionBlock

class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, language_backbone,box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.language_backbone = language_backbone
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        self.log_snr = logsnr_schedule_cosine
        embedding_dim = 768
        time_embedding_dim = 768
        self.num_sample_steps = 1
        self.time_embed = nn.Sequential(
            nn.Linear(time_embedding_dim, 4 * time_embedding_dim),
            nn.SiLU(),
            nn.Linear(4 * time_embedding_dim, time_embedding_dim),
        )
        resnet_block = partial(ResnetBlock, groups=8)
        self.down = nn.Sequential(
            ConvModule(in_channels=1, out_channels=embedding_dim, kernel_size=3, padding=1, stride=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            resnet_block(embedding_dim, embedding_dim, time_emb_dim=time_embedding_dim),
            ConvModule(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=3, padding=1, stride=1,
                       norm_cfg=dict(type='BN', requires_grad=True))
        )

        self.up = nn.Sequential(
            ConvModule(in_channels=embedding_dim * 2, out_channels=embedding_dim, kernel_size=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            # resnet_block(embedding_dim, embedding_dim),
            Upsample(embedding_dim, embedding_dim // 4, factor=2),
            ConvModule(in_channels=embedding_dim // 4, out_channels=embedding_dim // 4, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
            Upsample(embedding_dim // 4, embedding_dim // 8, factor=2),
            ConvModule(in_channels=embedding_dim // 8, out_channels=embedding_dim // 16, kernel_size=3, padding=1,
                       norm_cfg=dict(type='BN', requires_grad=True)),
        )
        self.pred = nn.Linear(768, 1)
        # self.upsample = torch.nn.Upsample(scale_factor=16, mode='nearest', align_corners=None)
        self.untied_text_pos_enc_p = Untied2DPositionalEncoder(dim=768, num_heads=8, h=6, w=6,with_q=False)
        self.untied_search_pos_enc_p = Untied2DPositionalEncoder(dim=768, num_heads=8, h=24, w=24,with_k=False)
        # self.rpe_index_p = generate_2d_concatenated_cross_attention_relative_positional_encoding_index(
        #     (8, 8), (16, 16))
        # self.rpe_bias_table_p = RelativePosition2DEncoder(hide_channel, self.rpe_index_p.max() + 1)
        self.cross_attention_p = CrossAttentionBlock(dim=768, num_heads=8)
        self.filter = SpatialAttention(dim=768,spatial_dim=1)


        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                text,
                search_mask=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                gt=None,
                search_seg=None,
                xx=None,
                logsnr=None
                ):
        if xx == None:
            img = search_mask
            groundtruth = search_seg
            """
            visual
            """

            # k = seg[5].unsqueeze(dim=0)
            # save_image(k , '../../visual/seg.png')
            # k = img[5].unsqueeze(dim=0)
            # save_image(k, '../../visual/img.png')
            # k = search[5]
            # save_image(k, '../../visual/search.png')
            # image = cv2.imread('../../visual/search.png')
            # gt = gt[5]
            # x,y,w,h = int(gt[0]*255),int(gt[1]*255),int(gt[2]*255),int(gt[3]*255)
            # cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.imwrite('../../visual/search_box.png',image)

            img = normalize_to_neg_one_to_one(img.unsqueeze(dim=1))

            # img = img.unsqueeze(dim=1)
            # seg = seg.unsqueeze(dim=1)
            times = torch.zeros((img.shape[0],), device=img.device).float().uniform_(0, 1)
            n_times = torch.zeros((img.shape[0],), device=img.device).float().uniform_(0, 1)
            xx ,logsnr  = self.noise_begin(x_start=img, times=n_times, seg=None)
            # k = xx[0]
            # save_image(k, '../../visual/seg_noise.png')
        # k = search[0]
        # save_image(k, '../visual/search.png')
        # k = xx[0]
        # save_image(k, '../visual/seg_noise.png')
        text_fea = self.language_backbone(text)
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        # feat_last = x
        # if isinstance(x, list):
        #     feat_last = x[-1]
        untied_text_pos_enc_k = self.untied_text_pos_enc_p()
        untied_search_pos_enc_p_q = self.untied_search_pos_enc_p()
        attn_pos_enc_p = (untied_search_pos_enc_p_q @ (untied_text_pos_enc_k.transpose(-2, -1))).unsqueeze(0)
        cat_feature = x[:, -self.feat_len_s:]
        segg = self.cross_attention_p(q=cat_feature, kv=text_fea,  q_ape=None, k_ape=None,
                                     attn_pos=attn_pos_enc_p) + cat_feature
        # segg = self.filter(segg)

        out = self.forward_head(cat_feature = segg, timesteps=logsnr, seg=xx,  gt_score_map=None)

        # k = out['seg_mask'].unsqueeze(dim=0)
        # k = self.upsample(k)
        # save_image(k, '../../seg.png')
        # k = search
        # save_image(k, '../../search.png')
        if self.training is True:
            if aux_dict['removed_indexes_s'][0] is not None:
                b = groundtruth.shape[0]
                device = groundtruth.device
                # u = groundtruth
                removed_indexes_cat = torch.cat(aux_dict['removed_indexes_s'], dim=1).tolist()
                groundtruth = groundtruth.view(b,576).to('cpu')
                groundtruth = groundtruth.scatter_(1,torch.LongTensor(removed_indexes_cat),0).view(b,24,24).to(device)
                # k = groundtruth
                # for i in range(b):
                #     if u[i].equal(k[i]) is False:
                #         s = u[i].unsqueeze(dim=0).unsqueeze(dim=0)
                #         ks = k[i].unsqueeze(dim=0).unsqueeze(dim=0)
                #         kk = self.upsample(ks)
                #         save_image(kk, '../../visual/seg.png')
                #         ss = self.upsample(s)
                #         save_image(ss, '../../visual/truth.png')
                #         ks = search[i]
                #         save_image(ks, '../../visual/search.png')
                #         gts = gt[i]
                #         x,y,w,h = int(gts[0]*256),int(gts[1]*256),int(gts[2]*256),int(gts[3]*256)
                #         image = cv2.imread('../../visual/truth.png')
                #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                #         cv2.imwrite('../../visual/search_box.png',image)
                #         print('yes')

            xx = out['seg_mask'].clamp_(0., 1.).unsqueeze(dim=1)
            xx = normalize_to_neg_one_to_one(xx)
            xx, logsnr = self.noise_begin(x_start=xx, times=times, seg=None)
            out2 = self.forward_head(cat_feature=segg, timesteps=logsnr, seg=xx, gt_score_map=None)

            return out,out2,groundtruth
        else:
            return out
    def noise_begin(self, x_start, times, seg=None, noise=None, *args, **kwargs):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x, log_snr = self.q_sample(x_start=x_start, times=times, noise=noise)
        return x ,log_snr

    def q_sample(self, x_start, times, noise = None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        log_snr = self.log_snr(times)

        log_snr_padded = right_pad_dims_to(x_start, log_snr)
        alpha, sigma = sqrt(log_snr_padded.sigmoid()), sqrt((-log_snr_padded).sigmoid())
        x_noised =  x_start * alpha + noise * sigma

        return x_noised, log_snr
    def forward_head(self, cat_feature,timesteps,seg,  gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """

        t = self.time_embed(timestep_embedding(timesteps, 768))
        # k = 0
        for blk in self.down:
            if isinstance(blk, ResnetBlock):
                seg = blk(seg, t)
            else:
                seg = blk(seg)
                # if k == 0:
                #     k=1
                #     seg = feature2token(seg)
                #     seg = self.cross_attention_p(q=seg, kv=text, q_ape=None, k_ape=None,
                #                                 attn_pos=attn_pos_enc_p)
                #     seg = token2feature(seg)

        c1 = token2feature(cat_feature)
        seg = torch.cat([c1, seg], dim=1)
        for blk in self.up:
            if isinstance(blk, ResnetBlock):
                seg = blk(seg, t)
            else:
                seg = blk(seg)
                # if k == 0:
                    # k=1
                    # seg = feature2token(seg)
                    # seg = self.cross_attention_p(q=seg, kv=text, q_ape=None, k_ape=None,
                    #                             attn_pos=attn_pos_enc_p)
                    # seg = token2feature(seg)
        B, _, _, _ = seg.shape
        fea_mask = seg.contiguous().view(B, 576, 768)
        fea_mask = self.filter(fea_mask)
        seg_mask = self.pred(fea_mask).view(B, 24, 24)
        enc_opt = fea_mask + cat_feature

        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map,
                   'seg_mask': seg_mask}
            return out
        else:
            raise NotImplementedError

    @torch.no_grad()
    def p_sample_loop(self,template ,search ,text, verbose=False, ce_template_mask=False):
        self.score_map = []
        self.size_map = []
        self.offset_map = []
        # img = torch.randn(shape, device=self.device)
        B, C , H, W = search.shape
        img = torch.randn((B,1,24,24) , device=search.device)
        steps = torch.linspace(1., 0., self.num_sample_steps + 1, device=search.device)

        for i in tqdm(range(self.num_sample_steps), desc='sampling loop time step', total=self.num_sample_steps,
                      disable=not verbose):
            times = steps[i]
            times_next = steps[i + 1]
            img = self.p_sample(i,img,text,template ,search , times, times_next)

        out = {'score_map': torch.stack(self.score_map).mean(dim=0),
                   'size_map': torch.stack(self.size_map).mean(dim=0),
                   'offset_map': torch.stack(self.offset_map).mean(dim=0) }
        return out

    @torch.no_grad()
    def p_sample(self, i,x,text,template ,search , time, time_next):
        batch, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(i=i ,x=x,text=text, template=template, search=search, time=time,
                                                          time_next=time_next)

        if time_next == 0:
            return model_mean

        noise = torch.randn_like(x)
        return model_mean + sqrt(model_variance) * noise

    def p_mean_variance(self, i, x, text,template ,search , time, time_next):

        log_snr = self.log_snr(time)
        log_snr_next = self.log_snr(time_next)
        c = -expm1(log_snr - log_snr_next)

        squared_alpha, squared_alpha_next = log_snr.sigmoid(), log_snr_next.sigmoid()
        squared_sigma, squared_sigma_next = (-log_snr).sigmoid(), (-log_snr_next).sigmoid()

        alpha, sigma, alpha_next = map(sqrt, (squared_alpha, squared_sigma, squared_alpha_next))

        batch_log_snr = repeat(log_snr, ' -> b', b=x.shape[0])
        pred = self.forward(text=text,template=template,search=search, xx=x,logsnr=batch_log_snr)
        x_start = pred['seg_mask'].tanh()
            # x_start = x

        x_start.clamp_(-1., 1.)
        if i < 10 :
            self.score_map.append(pred['score_map']) # change to pred when generate cam
            self.size_map.append(pred['size_map'])
            self.offset_map.append(pred['offset_map'])

        model_mean = alpha_next * (x * (1 - c) / alpha + c * x_start)

        posterior_variance = squared_sigma_next * c

        return model_mean, posterior_variance
def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)
    language_backbone = build_bert()
    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        language_backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('missing_keys')
        print(missing_keys)
        print('unexpercted_keys')
        print(unexpected_keys)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

def logsnr_schedule_cosine(t, logsnr_min = -15, logsnr_max = 15):
    t_min = math.atan(math.exp(-0.5 * logsnr_max))
    t_max = math.atan(math.exp(-0.5 * logsnr_min))
    return -2 * log(torch.tan(t_min + t * (t_max - t_min)))
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def token2feature(tokens):
    B,L,D=tokens.shape
    H=W=int(L**0.5)
    x = tokens.permute(0, 2, 1).view(B, D, W, H).contiguous()
    return x


def feature2token(x):
    B,C,W,H = x.shape
    L = W*H
    tokens = x.view(B, C, L).permute(0, 2, 1).contiguous()
    return tokens


def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm1d(out_planes),
        nn.ReLU(inplace=True))

class SpatialAttention(nn.Module):
    def __init__(self, dim, num_heads=8, spatial_dim=1, qkv_bias=False, attn_drop=0., proj_drop=0., kernel_size=3):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.spatial_dim = spatial_dim
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        # self.summation = nn.Conv2d(dim, spatial_dim, kernel_size=kernel_size, padding=1)
        # if kernel_size is 3:
        #     self.summation = nn.Conv2d(dim, spatial_dim, kernel_size=kernel_size, padding=1)
        # elif kernel_size is 1:
        #     self.summation = nn.Conv2d(dim, spatial_dim, kernel_size=kernel_size)
        self.summation = nn.Linear(dim,spatial_dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm = nn.BatchNorm1d(576)
        self.cross_heads = nn.Linear(self.spatial_dim, self.spatial_dim, bias=qkv_bias)

    def forward(self, x, prompts=None, return_token=False):
        # x: Bx, Cx, W, H
        # mask: [B, N, ] torch.bool
        Bx, N ,Cx = x.shape
        space_attn = self.summation(x)
        x_expanded = x.view(Bx, Cx, -1).transpose(1, 2).unsqueeze(-2)
        space_attn = space_attn.view(Bx, self.spatial_dim, -1).transpose(1, 2)
        space_attn_expanded = space_attn.softmax(dim=1).unsqueeze(-1).expand(Bx, -1, self.spatial_dim, Cx)
        spatial_x = space_attn_expanded * x_expanded
        # attn_topk
        tokens = torch.topk(spatial_x, 1, dim=1, largest=True)[0].squeeze(1)
        qkv = self.qkv(tokens).reshape(Bx, self.spatial_dim, 3, self.num_heads, Cx // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_spatial_x = (attn @ v).transpose(1, 2).reshape(Bx, -1, Cx)
        attn_spatial_x = self.proj_drop(attn_spatial_x).transpose(1, 2)
        attn_spatial_x = self.cross_heads(attn_spatial_x).transpose(1, 2)

        out = (space_attn @ attn_spatial_x).reshape(Bx, N, Cx).permute(0,1, 2).contiguous()
        out = self.norm(out) + x
        return out