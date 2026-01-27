# -*- coding: utf-8 -*-
# Author: Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import os
import sys
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from opencood.models.sub_modules.lss_submodule import Up, CamEncode, BevEncode, CamEncode_Resnet101
from opencood.utils.camera_utils import gen_dx_bx, cumsum_trick, QuickCumsum, depth_discretization, bin_depths
from opencood.models.sub_modules.pillar_vfe import PillarVFE
from opencood.models.sub_modules.point_pillar_scatter import PointPillarScatter
from opencood.models.sub_modules.base_bev_backbone_resnet import ResNetBEVBackbone
from opencood.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from opencood.models.sub_modules.downsample_conv import DownsampleConv
from opencood.models.sub_modules.mean_vfe import MeanVFE
from opencood.models.sub_modules.height_compression import HeightCompression



class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()
        grid_size = (np.array(args['lidar_range'][3:6]) - np.array(args['lidar_range'][0:3])) / \
                            np.array(args['voxel_size'])
        grid_size = np.round(grid_size).astype(np.int64)
        args['point_pillar_scatter']['grid_size'] = grid_size

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args['pillar_vfe'],
                                    num_point_features=4,
                                    voxel_size=args['voxel_size'],
                                    point_cloud_range=args['lidar_range'])
        self.scatter = PointPillarScatter(args['point_pillar_scatter'])


    def forward(self, data_dict, modality_name):
        voxel_features = data_dict[f'inputs_{modality_name}']['voxel_features']
        voxel_coords = data_dict[f'inputs_{modality_name}']['voxel_coords']
        voxel_num_points = data_dict[f'inputs_{modality_name}']['voxel_num_points']
    
        batch_dict = {'voxel_features': voxel_features,
                      'voxel_coords': voxel_coords,
                      'voxel_num_points': voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        lidar_feature_2d = batch_dict['spatial_features'] # H0, W0
        return lidar_feature_2d

class SECOND(nn.Module):
    def __init__(self, args):
        super(SECOND, self).__init__()
        # Import spconv backbone lazily to avoid requiring spconv for camera-only runs.
        from opencood.models.sub_modules.sparse_backbone_3d import VoxelBackBone8x
        lidar_range = np.array(args['lidar_range'])
        grid_size = np.round((lidar_range[3:6] - lidar_range[:3]) /
                                np.array(args['voxel_size'])).astype(np.int64)
        self.vfe = MeanVFE(args['mean_vfe'],
                            args['mean_vfe']['num_point_features'])
        self.spconv_block = VoxelBackBone8x(args['spconv'],
                                            input_channels=args['spconv'][
                                                'num_features_in'],
                                            grid_size=grid_size)
        self.map_to_bev = HeightCompression(args['map2bev'])

    def forward(self, data_dict, modality_name):
        voxel_features = data_dict[f'inputs_{modality_name}']['voxel_features']
        voxel_coords = data_dict[f'inputs_{modality_name}']['voxel_coords']
        voxel_num_points = data_dict[f'inputs_{modality_name}']['voxel_num_points']
        batch_size = voxel_coords[:,0].max() + 1


        batch_dict = {'voxel_features': voxel_features,
                    'voxel_coords': voxel_coords,
                    'voxel_num_points': voxel_num_points,
                    'batch_size': batch_size}

        batch_dict = self.vfe(batch_dict)
        batch_dict = self.spconv_block(batch_dict)
        batch_dict = self.map_to_bev(batch_dict)
        return batch_dict['spatial_features']

class LiftSplatShoot(nn.Module):
    def __init__(self, args): 
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = args['grid_conf']   # 网格配置参数
        self.data_aug_conf = args['data_aug_conf']   # 数据增强配置参数
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                self.grid_conf['ybound'],
                                self.grid_conf['zbound'],
                                )  # 划分网格

        self.dx = dx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [0.4,0.4,20]
        self.bx = bx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [-49.8,-49.8,0]
        self.nx = nx.clone().detach().requires_grad_(False).to(torch.device("cuda"))  # [250,250,1]
        self.depth_supervision = args['depth_supervision']
        self.downsample = args['img_downsample']  # 下采样倍数
        self.camC = args['img_features']  # 图像特征维度
        self.frustum = self.create_frustum().clone().detach().requires_grad_(False).to(torch.device("cuda"))  # frustum: DxfHxfWx3(41x8x16x3)
        self.use_quickcumsum = True
        self.D, _, _, _ = self.frustum.shape  # D: 41
        self.camera_encoder_type = args['camera_encoder']
        if self.camera_encoder_type == 'EfficientNet':
            self.camencode = CamEncode(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
        elif self.camera_encoder_type == 'Resnet101':
            self.camencode = CamEncode_Resnet101(self.D, self.camC, self.downsample, \
                self.grid_conf['ddiscr'], self.grid_conf['mode'], args['use_depth_gt'], args['depth_supervision'])
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 原始图片大小  ogfH:128  ogfW:288
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 下采样16倍后图像大小  fH: 12  fW: 22
        # ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)  # 在深度方向上划分网格 ds: DxfHxfW(41x12x22)
        ds = torch.tensor(depth_discretization(*self.grid_conf['ddiscr'], self.grid_conf['mode']), dtype=torch.float).view(-1,1,1).expand(-1, fH, fW)

        D, _, _ = ds.shape # D: 41 表示深度方向上网格的数量
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)  # 在0到288上划分18个格子 xs: DxfHxfW(41x12x22)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)  # 在0到127上划分8个格子 ys: DxfHxfW(41x12x22)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)  # 堆积起来形成网格坐标, frustum[i,j,k,0]就是(i,j)位置，深度为k的像素的宽度方向上的栅格坐标   frustum: DxfHxfWx3
        return frustum

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _ = trans.shape  # B:4(batchsize)    N: 4(相机数目)

        # undo post-transformation
        # B x N x D x H x W x 3
        # 抵消数据增强及预处理对像素的变化
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],  # points[:, :, :, :, :, 2:3] ranges from [4, 45) meters
                            points[:, :, :, :, :, 2:3]
                            ), 5)  # 将像素坐标(u,v,d)变成齐次坐标(du,dv,d)
        # d[u,v,1]^T=intrins*rots^(-1)*([x,y,z]^T-trans)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)  # 将像素坐标d[u,v,1]^T转换到车体坐标系下的[x,y,z]
        
        return points  # B x N x D x H x W x 3 (4 x 4 x 41 x 16 x 22 x 3) 

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape  # B: 4  N: 4  C: 3  imH: 256  imW: 352

        x = x.view(B*N, C, imH, imW)  # B和N两个维度合起来  x: 16 x 4 x 256 x 352
        depth_items, x = self.camencode(x) # 进行图像编码  x: B*N x C x D x fH x fW(24 x 64 x 41 x 16 x 22)
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)  #将前两维拆开 x: B x N x C x D x fH x fW(4 x 6 x 64 x 41 x 16 x 22)
        x = x.permute(0, 1, 3, 4, 5, 2)  # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64)

        return x, depth_items

    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)  # 消除掉z维

        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)  # 像素坐标到自车中坐标的映射关系 geom: B x N x D x H x W x 3 (4 x N x 42 x 16 x 22 x 3)
        x_img, depth_items = self.get_cam_feats(x)  # 提取图像特征并预测深度编码 x: B x N x D x fH x fW x C(4 x N x 42 x 16 x 22 x 64)
        x = self.voxel_pooling(geom, x_img)  # x: 4 x 64 x 240 x 240

        return x, depth_items

    def forward(self, data_dict, modality_name):
        # x: [4,4,3,256, 352]
        # rots: [4,4,3,3]
        # trans: [4,4,3]
        # intrins: [4,4,3,3]
        # post_rots: [4,4,3,3]
        # post_trans: [4,4,3]
        image_inputs_dict = data_dict[f'inputs_{modality_name}']
        x, rots, trans, intrins, post_rots, post_trans = \
            image_inputs_dict['imgs'], image_inputs_dict['rots'], image_inputs_dict['trans'], image_inputs_dict['intrins'], image_inputs_dict['post_rots'], image_inputs_dict['post_trans']
        x, depth_items = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)  # 将图像转换到BEV下，x: B x C x 240 x 240 (4 x 64 x 240 x 240)
        
        if self.depth_supervision:
            self.depth_items = depth_items

        return x


class LiftSplatShootVoxel(LiftSplatShoot):
    def voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3 (4 x 6 x 41 x 16 x 22 x 3), D is discretization in "UD" or "LID"
        # x: B x N x D x fH x fW x C(4 x 6 x 41 x 16 x 22 x 64), D is num_bins

        B, N, D, H, W, C = x.shape  # B: 4  N: 6  D: 41  H: 16  W: 22  C: 64
        Nprime = B*N*D*H*W  # Nprime

        # flatten x
        x = x.reshape(Nprime, C)  # 将图像展平，一共有 B*N*D*H*W 个点

        # flatten indices

        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()  # 将[-48,48] [-10 10]的范围平移到 [0, 240), [0, 1) 计算栅格坐标并取整
        geom_feats = geom_feats.view(Nprime, 3)  # 将像素映射关系同样展平  geom_feats: B*N*D*H*W x 3 
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])  # 每个点对应于哪个batch
        geom_feats = torch.cat((geom_feats, batch_ix), 1)  # geom_feats: B*N*D*H*W x 4, geom_feats[:,3]表示batch_id

        # filter out points that are outside box
        # 过滤掉在边界线之外的点 x:0~240  y: 0~240  z: 0
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept] 
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]  # 给每一个点一个rank值，rank相等的点在同一个batch，并且在在同一个格子里面
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]  # 按照rank排序，这样rank相近的点就在一起了
        # x: 168648 x 64  geom_feats: 168648 x 4  ranks: 168648

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)  # 一个batch的一个格子里只留一个点 x: 29072 x 64  geom_feats: 29072 x 4

        # griddify (B x C x Z x X x Y)
        # final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)  # final: 4 x 64 x Z x X x Y
        # final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x  # 将x按照栅格坐标放到final中

        # modify griddify (B x C x Z x Y x X) by Yifan Lu 2022.10.7
        # ------> x
        # |
        # |
        # y
        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)  # final: 4 x 64 x Z x Y x X
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x  # 将x按照栅格坐标放到final中

        # collapse Z
        #final = torch.max(final.unbind(dim=2), 1)[0]  # 消除掉z维
        final = torch.max(final, 2)[0]  # 消除掉z维
        return final  # final: 4 x 64 x 240 x 240  # B, C, H, W 


class DPTLiftSplatShoot(LiftSplatShoot):
    def __init__(self, args):
        super(DPTLiftSplatShoot, self).__init__(args)
        self.dpt_root = args.get("dpt_root", None)
        self.dpt_model_name = args.get("dpt_model_name", "da3-large")
        self.dpt_pretrained = args.get("dpt_pretrained", None)
        self.dpt_freeze = args.get("dpt_freeze", True)
        self.dpt_freeze_backbone_blocks = int(args.get("dpt_freeze_backbone_blocks", 0) or 0)
        self.dpt_freeze_patch_embed = bool(args.get("dpt_freeze_patch_embed", True))
        self.dpt_freeze_pos_embed = bool(args.get("dpt_freeze_pos_embed", True))
        self.dpt_use_amp = args.get("dpt_use_amp", True)
        self.dpt_amp_dtype = args.get("dpt_amp_dtype", "bf16")
        self.dpt_patch_size = int(args.get("dpt_patch_size", 14))
        self.dpt_resize = args.get("dpt_resize", None)
        self.depth_min = args.get("depth_min", 0.1)
        self.depth_max = args.get("depth_max", 80.0)
        self.depth_prior_mode = args.get("depth_prior_mode", "none")

        # Override camera encoder to consume DPT depth as GT depth
        self.depth_supervision = False
        self.use_gt_depth = True
        self.camera_encoder_type = args.get("camera_encoder", "EfficientNet")
        if self.camera_encoder_type == "EfficientNet":
            self.camencode = CamEncode(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                use_gt_depth=True,
                depth_supervision=False,
            )
        elif self.camera_encoder_type == "Resnet101":
            self.camencode = CamEncode_Resnet101(
                self.D,
                self.camC,
                self.downsample,
                self.grid_conf["ddiscr"],
                self.grid_conf["mode"],
                use_gt_depth=True,
                depth_supervision=False,
            )

        self._init_dpt()

    def _build_depth_dist(self, depth, fH, fW):
        # depth: B x N x H x W
        B, N, H, W = depth.shape
        if H != fH or W != fW:
            depth = F.interpolate(depth.view(B * N, 1, H, W), size=(fH, fW), mode="bilinear", align_corners=False)
            depth = depth.view(B, N, fH, fW)
        depth = depth.clamp(min=self.depth_min, max=self.depth_max)
        depth_flat = depth.view(B * N, fH, fW)
        depth_indices, mask = bin_depths(
            depth_flat,
            self.grid_conf["mode"],
            self.grid_conf["ddiscr"][0],
            self.grid_conf["ddiscr"][1],
            self.grid_conf["ddiscr"][2],
            target=self.training,
        )
        depth_dist = F.one_hot(depth_indices.long(), num_classes=self.grid_conf["ddiscr"][2]).permute(0, 3, 1, 2).float()
        if not self.training:
            mask = mask.unsqueeze(1)
            depth_dist = depth_dist * mask
        return depth_dist

    def _compute_depth_prior_cam(self, depth, depth_dist, fH, fW):
        if self.depth_prior_mode in (None, "none", "off"):
            return None
        if depth.shape[-2:] != (fH, fW):
            depth = F.interpolate(depth.view(-1, 1, depth.shape[-2], depth.shape[-1]), size=(fH, fW), mode="bilinear", align_corners=False)
            depth = depth.view(depth_dist.shape[0], fH, fW)
        if self.depth_prior_mode == "inverse_depth":
            conf = (self.depth_max - depth) / max(self.depth_max - self.depth_min, 1e-6)
            conf = conf.clamp(min=0.0, max=1.0)
        elif self.depth_prior_mode == "valid_mask":
            conf = (depth > 0).float()
        else:
            raise ValueError(f"Unknown depth_prior_mode: {self.depth_prior_mode}")
        conf = conf.view(depth_dist.shape[0], 1, fH, fW)
        depth_prior = depth_dist * conf
        # depth_dist is (B*N, D, fH, fW)
        BN = depth_dist.shape[0]
        return depth_prior.view(BN, self.D, fH, fW)

    def _init_dpt(self):
        if importlib.util.find_spec("depth_anything_3") is None:
            dpt_root = self.dpt_root
            if dpt_root is None:
                dpt_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../../../Depth-Anything-3/src")
                )
            if os.path.isdir(dpt_root):
                sys.path.insert(0, dpt_root)

        from depth_anything_3.api import DepthAnything3

        if self.dpt_pretrained:
            self.dpt = DepthAnything3.from_pretrained(self.dpt_pretrained)
        else:
            self.dpt = DepthAnything3(model_name=self.dpt_model_name)

        self.dpt.eval()
        if self.dpt_freeze:
            for p in self.dpt.parameters():
                p.requires_grad_(False)
        elif self.dpt_freeze_backbone_blocks > 0:
            self._freeze_dpt_backbone(self.dpt_freeze_backbone_blocks)

    def _freeze_dpt_backbone(self, num_blocks: int):
        try:
            backbone = getattr(self.dpt, "model", self.dpt).backbone
            pretrained = getattr(backbone, "pretrained", backbone)
            blocks = getattr(pretrained, "blocks", None)
        except Exception:
            return

        if blocks is not None:
            for idx, block in enumerate(blocks):
                if idx >= num_blocks:
                    break
                for p in block.parameters():
                    p.requires_grad_(False)

        if self.dpt_freeze_patch_embed and hasattr(pretrained, "patch_embed"):
            for p in pretrained.patch_embed.parameters():
                p.requires_grad_(False)
        if self.dpt_freeze_pos_embed:
            for name in ("pos_embed", "cls_token", "camera_token"):
                if hasattr(pretrained, name):
                    getattr(pretrained, name).requires_grad_(False)

    def _dpt_amp_dtype(self):
        if self.dpt_amp_dtype == "fp16":
            return torch.float16
        if self.dpt_amp_dtype == "bf16":
            return torch.bfloat16
        return torch.float32

    def _resize_for_dpt(self, imgs):
        B, N, C, H, W = imgs.shape
        if self.dpt_resize is not None:
            target_h, target_w = self.dpt_resize
        else:
            target_h = (H // self.dpt_patch_size) * self.dpt_patch_size
            target_w = (W // self.dpt_patch_size) * self.dpt_patch_size
            target_h = max(target_h, self.dpt_patch_size)
            target_w = max(target_w, self.dpt_patch_size)
        if target_h == H and target_w == W:
            return imgs, (H, W)
        imgs_flat = imgs.view(B * N, C, H, W)
        imgs_flat = F.interpolate(imgs_flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return imgs_flat.view(B, N, C, target_h, target_w), (H, W)

    @staticmethod
    def _ensure_rgb(imgs):
        # DepthAnything expects 3-channel RGB input. Drop extra channels (e.g., alpha).
        if imgs.shape[2] > 3:
            return imgs[:, :, :3, ...]
        return imgs

    def _predict_depth(self, imgs):
        imgs = self._ensure_rgb(imgs)
        if next(self.dpt.parameters()).device != imgs.device:
            self.dpt.to(imgs.device)
        imgs_dpt, (H, W) = self._resize_for_dpt(imgs)
        device_type = "cuda" if imgs.device.type == "cuda" else "cpu"
        use_amp = self.dpt_use_amp and device_type == "cuda"
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=use_amp, dtype=self._dpt_amp_dtype()):
                raw = self.dpt(imgs_dpt, extrinsics=None, intrinsics=None, export_feat_layers=[], infer_gs=False)
        depth = raw["depth"]  # (B, N, 1, h, w)
        if depth.shape[-2:] != (H, W):
            depth_flat = depth.view(depth.shape[0] * depth.shape[1], 1, depth.shape[-2], depth.shape[-1])
            depth_flat = F.interpolate(depth_flat, size=(H, W), mode="bilinear", align_corners=False)
            depth = depth_flat.view(depth.shape[0], depth.shape[1], 1, H, W)
        depth = depth.clamp(min=self.depth_min, max=self.depth_max)
        return depth.squeeze(2)

    def get_cam_feats(self, x):
        B, N, _, imH, imW = x.shape
        x_rgb = self._ensure_rgb(x)
        depth = self._predict_depth(x_rgb)
        x = torch.cat([x_rgb, depth.unsqueeze(2)], dim=2)
        x = x.view(B * N, 4, imH, imW)
        depth_items, x = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)
        fH, fW = imH // self.downsample, imW // self.downsample
        self._depth_prior_cam = None
        if self.depth_prior_mode not in (None, "none", "off"):
            depth_dist = self._build_depth_dist(depth, fH, fW)
            depth_prior = self._compute_depth_prior_cam(depth, depth_dist, fH, fW)
            if depth_prior is not None:
                self._depth_prior_cam = depth_prior.view(B, N, self.D, fH, fW, 1)
        return x, depth_items

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x_img, depth_items = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x_img)
        if hasattr(self, "_depth_prior_cam") and self._depth_prior_cam is not None:
            depth_prior = self.voxel_pooling(geom, self._depth_prior_cam)
            self.depth_prior = depth_prior
        return x, depth_items


class MapAnythingTokenBEV(nn.Module):
    def __init__(self, args):
        super(MapAnythingTokenBEV, self).__init__()
        self.args = args

        self.mapanything_root = args.get("mapanything_root", None)
        self.mapanything_config_path = args.get("mapanything_config_path", None)
        self.mapanything_model_config = args.get("mapanything_model_config", None)
        self.mapanything_model_str = args.get("mapanything_model_str", "mapanything")
        self.torch_hub_force_reload = args.get("torch_hub_force_reload", False)
        self.freeze_mapanything = args.get("freeze_mapanything", True)

        self.use_amp = args.get("use_amp", False)
        self.amp_dtype = args.get("amp_dtype", "bf16")
        self.memory_efficient_inference = args.get("memory_efficient_inference", False)

        self.use_optional_geom_inputs = args.get("use_optional_geom_inputs", True)
        self.use_intrinsics = args.get("use_intrinsics", True)
        self.use_depth_input = args.get("use_depth_input", False)
        self.use_post_transform = args.get("use_post_transform", False)

        self.use_confidence_mask = args.get("use_confidence_mask", False)
        self.confidence_threshold = args.get("confidence_threshold", 0.5)
        self.use_mask = args.get("use_mask", False)

        self.depth_min = args.get("depth_min", 0.1)
        self.depth_max = args.get("depth_max", 80.0)

        self.grid_conf = args["grid_conf"]
        dx, bx, nx = gen_dx_bx(self.grid_conf["xbound"],
                               self.grid_conf["ybound"],
                               self.grid_conf["zbound"])
        self.register_buffer("dx", dx.clone().detach().requires_grad_(False))
        self.register_buffer("bx", bx.clone().detach().requires_grad_(False))
        self.register_buffer("nx", nx.clone().detach().requires_grad_(False))

        self._init_mapanything()

        out_channels = args.get("out_channels", self.mapanything.info_sharing.dim)
        in_channels = self.mapanything.info_sharing.dim
        self.feat_proj = None
        if out_channels != in_channels:
            self.feat_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def _init_mapanything(self):
        if importlib.util.find_spec("mapanything") is None:
            map_root = self.mapanything_root
            if map_root is None:
                map_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../../../map-anything")
                )
            if os.path.isdir(map_root):
                sys.path.insert(0, map_root)

        from omegaconf import OmegaConf
        from mapanything.models import init_model

        if self.mapanything_model_config is None:
            if self.mapanything_config_path is None:
                raise ValueError("mapanything_config_path or mapanything_model_config must be provided.")
            config_path = self.mapanything_config_path
            if config_path.endswith(".json"):
                import json
                with open(config_path, "r") as f:
                    config_data = json.load(f)
            else:
                import yaml
                with open(config_path, "r") as f:
                    config_data = yaml.safe_load(f)
            if "model_str" in config_data and "model_config" in config_data:
                model_str = config_data["model_str"]
                model_config = config_data["model_config"]
            else:
                model_str = self.mapanything_model_str
                model_config = config_data
        else:
            model_str = self.mapanything_model_str
            model_config = self.mapanything_model_config

        model_cfg = OmegaConf.create(model_config)
        self.mapanything = init_model(model_str, model_cfg, torch_hub_force_reload=self.torch_hub_force_reload)

        self.data_norm_type = self.mapanything.encoder_config.get("data_norm_type", "dinov2")

        if self.freeze_mapanything:
            self.mapanything.eval()
            for p in self.mapanything.parameters():
                p.requires_grad_(False)

    def _amp_dtype(self):
        if self.amp_dtype == "fp16":
            return torch.float16
        if self.amp_dtype == "bf16":
            return torch.bfloat16
        return torch.float32

    def _build_views(self, image_inputs_dict):
        imgs = image_inputs_dict["imgs"]
        _, num_views, _, _, _ = imgs.shape
        views = []
        for view_idx in range(num_views):
            view = {
                "img": imgs[:, view_idx],
                "data_norm_type": [self.data_norm_type],
            }
            if self.use_intrinsics and "intrins" in image_inputs_dict:
                view["intrinsics"] = image_inputs_dict["intrins"][:, view_idx]
            if self.use_depth_input and "depths" in image_inputs_dict:
                view["depth_z"] = image_inputs_dict["depths"][:, view_idx]
            views.append(view)
        if self.use_optional_geom_inputs:
            from mapanything.utils.inference import preprocess_input_views_for_inference
            views = preprocess_input_views_for_inference(views)
        return views

    def _get_geometry_from_depth(self, depth, rots, trans, intrins, post_rots=None, post_trans=None):
        # depth: B x N x 1 x H x W
        B, N, _, H, W = depth.shape
        device = depth.device
        xs = torch.linspace(0, W - 1, W, device=device, dtype=depth.dtype)
        ys = torch.linspace(0, H - 1, H, device=device, dtype=depth.dtype)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        points = torch.stack((grid_x, grid_y, torch.ones_like(grid_x)), dim=-1)
        points = points.view(1, 1, H, W, 3).repeat(B, N, 1, 1, 1)

        if post_rots is not None and post_trans is not None:
            points = points - post_trans.view(B, N, 1, 1, 3)
            points = torch.inverse(post_rots).view(B, N, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)

        depth = depth.permute(0, 1, 3, 4, 2)
        points = torch.cat((points[..., :2] * depth, depth), dim=-1)

        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 3, 3).matmul(points.unsqueeze(-1)).squeeze(-1)
        points = points + trans.view(B, N, 1, 1, 3)
        return points

    def _voxel_pooling(self, geom_feats, x):
        # geom_feats: B x N x D x H x W x 3
        # x: B x N x D x H x W x C
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        x = x.reshape(Nprime, C)

        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
            + geom_feats[:, 1] * (self.nx[2] * B) \
            + geom_feats[:, 2] * B \
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x
        final = torch.max(final, 2)[0]
        return final

    def forward(self, data_dict, modality_name):
        image_inputs_dict = data_dict[f"inputs_{modality_name}"]
        imgs = image_inputs_dict["imgs"]
        rots = image_inputs_dict["rots"]
        trans = image_inputs_dict["trans"]
        intrins = image_inputs_dict["intrins"]
        post_rots = image_inputs_dict.get("post_rots", None)
        post_trans = image_inputs_dict.get("post_trans", None)

        views = self._build_views(image_inputs_dict)

        if self.freeze_mapanything:
            self.mapanything.eval()

        with torch.autocast("cuda", enabled=self.use_amp, dtype=self._amp_dtype()):
            preds, feats = self.mapanything(
                views,
                memory_efficient_inference=self.memory_efficient_inference,
                return_features=True,
            )

        feat_list = feats["final_features"]
        feat = torch.stack(feat_list, dim=1)
        B, N, C, Hf, Wf = feat.shape

        if self.feat_proj is not None:
            feat = feat.view(B * N, C, Hf, Wf)
            feat = self.feat_proj(feat)
            feat = feat.view(B, N, self.out_channels, Hf, Wf)

        depth_list = []
        conf_list = []
        mask_list = []
        for pred in preds:
            if "depth_along_ray" in pred:
                depth = pred["depth_along_ray"]
            elif "depth_z" in pred:
                depth = pred["depth_z"]
            elif "pts3d_cam" in pred:
                depth = torch.norm(pred["pts3d_cam"], dim=-1, keepdim=True)
            else:
                raise ValueError("MapAnything output does not contain depth.")
            depth_list.append(depth.permute(0, 3, 1, 2))
            if "conf" in pred:
                conf_list.append(pred["conf"].unsqueeze(1))
            if "mask" in pred:
                mask_list.append(pred["mask"].float().unsqueeze(1))

        depth = torch.stack(depth_list, dim=1)
        depth = depth.clamp(min=self.depth_min, max=self.depth_max)

        H, W = depth.shape[-2:]
        if H != Hf or W != Wf:
            depth = F.interpolate(depth.view(B * N, 1, H, W), size=(Hf, Wf), mode="bilinear", align_corners=False)
            depth = depth.view(B, N, 1, Hf, Wf)

        if self.use_confidence_mask and conf_list:
            conf = torch.stack(conf_list, dim=1)
            if conf.shape[-2:] != (Hf, Wf):
                conf = F.interpolate(conf.view(B * N, 1, H, W), size=(Hf, Wf), mode="bilinear", align_corners=False)
                conf = conf.view(B, N, 1, Hf, Wf)
            feat = feat * (conf >= self.confidence_threshold).to(feat.dtype)

        if self.use_mask and mask_list:
            mask = torch.stack(mask_list, dim=1)
            if mask.shape[-2:] != (Hf, Wf):
                mask = F.interpolate(mask.view(B * N, 1, H, W), size=(Hf, Wf), mode="nearest")
                mask = mask.view(B, N, 1, Hf, Wf)
            feat = feat * mask.to(feat.dtype)

        scale_x = Wf / float(W)
        scale_y = Hf / float(H)
        intrins_scaled = intrins.clone()
        intrins_scaled[..., 0, 0] *= scale_x
        intrins_scaled[..., 1, 1] *= scale_y
        intrins_scaled[..., 0, 2] *= scale_x
        intrins_scaled[..., 1, 2] *= scale_y

        post_rots_scaled = None
        post_trans_scaled = None
        if self.use_post_transform and post_rots is not None and post_trans is not None:
            post_rots_scaled = post_rots.clone()
            post_trans_scaled = post_trans.clone()
            post_trans_scaled[..., 0] *= scale_x
            post_trans_scaled[..., 1] *= scale_y

        geom = self._get_geometry_from_depth(
            depth,
            rots.to(device=depth.device, dtype=depth.dtype),
            trans.to(device=depth.device, dtype=depth.dtype),
            intrins_scaled.to(device=depth.device, dtype=depth.dtype),
            post_rots=post_rots_scaled.to(device=depth.device, dtype=depth.dtype)
            if post_rots_scaled is not None
            else None,
            post_trans=post_trans_scaled.to(device=depth.device, dtype=depth.dtype)
            if post_trans_scaled is not None
            else None,
        )

        feat = feat.permute(0, 1, 3, 4, 2).unsqueeze(2)
        geom = geom.unsqueeze(2)
        bev = self._voxel_pooling(geom, feat)

        return bev


class DPTTransformerLiftSplatShoot(LiftSplatShoot):
    def __init__(self, args):
        super(DPTTransformerLiftSplatShoot, self).__init__(args)
        self.dpt_root = args.get("dpt_root", None)
        self.dpt_model_name = args.get("dpt_model_name", "da3metric-large")
        self.dpt_pretrained = args.get("dpt_pretrained", None)
        self.dpt_freeze = args.get("dpt_freeze", True)
        self.dpt_freeze_backbone_blocks = int(args.get("dpt_freeze_backbone_blocks", 0) or 0)
        self.dpt_freeze_patch_embed = bool(args.get("dpt_freeze_patch_embed", True))
        self.dpt_freeze_pos_embed = bool(args.get("dpt_freeze_pos_embed", True))
        self.dpt_use_amp = args.get("dpt_use_amp", True)
        self.dpt_amp_dtype = args.get("dpt_amp_dtype", "bf16")
        self.dpt_patch_size = int(args.get("dpt_patch_size", 14))
        self.dpt_resize = args.get("dpt_resize", None)
        self.dpt_feat_layers = args.get("dpt_feat_layers", [11])
        if isinstance(self.dpt_feat_layers, int):
            self.dpt_feat_layers = [self.dpt_feat_layers]
        self.dpt_feat_dim = args.get("dpt_feat_dim", None)
        self.dpt_feat_fuse = args.get("dpt_feat_fuse", "concat").lower()
        self.dpt_feat_fuse = self.dpt_feat_fuse.replace("-", "_")

        self.depth_min = args.get("depth_min", 0.1)
        self.depth_max = args.get("depth_max", 80.0)
        self.depth_prior_mode = args.get("depth_prior_mode", "none")

        self.feat_proj = None
        self.layer_proj = None
        self.layer_weights = None
        self.fuse_conv = None
        if self.dpt_feat_fuse in ("weighted_sum", "pyramid"):
            self.layer_weights = nn.Parameter(torch.zeros(len(self.dpt_feat_layers)))
        if self.dpt_feat_fuse in ("sum", "weighted_sum", "pyramid"):
            if self.dpt_feat_dim is not None:
                self.layer_proj = nn.ModuleList(
                    [nn.Conv2d(self.dpt_feat_dim, self.camC, kernel_size=1) for _ in self.dpt_feat_layers]
                )
            else:
                self.layer_proj = nn.ModuleList(
                    [nn.LazyConv2d(self.camC, kernel_size=1) for _ in self.dpt_feat_layers]
                )
        elif self.dpt_feat_fuse == "concat":
            if self.dpt_feat_dim is not None and len(self.dpt_feat_layers) == 1:
                self.feat_proj = nn.Conv2d(self.dpt_feat_dim, self.camC, kernel_size=1)
            else:
                self.feat_proj = nn.LazyConv2d(self.camC, kernel_size=1)

        self._init_dpt()

    def _init_dpt(self):
        if importlib.util.find_spec("depth_anything_3") is None:
            dpt_root = self.dpt_root
            if dpt_root is None:
                dpt_root = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../../../Depth-Anything-3/src")
                )
            if os.path.isdir(dpt_root):
                sys.path.insert(0, dpt_root)

        from depth_anything_3.api import DepthAnything3

        if self.dpt_pretrained:
            self.dpt = DepthAnything3.from_pretrained(self.dpt_pretrained)
        else:
            self.dpt = DepthAnything3(model_name=self.dpt_model_name)

        self.dpt.eval()
        if self.dpt_freeze:
            for p in self.dpt.parameters():
                p.requires_grad_(False)
        elif self.dpt_freeze_backbone_blocks > 0:
            self._freeze_dpt_backbone(self.dpt_freeze_backbone_blocks)

    def _freeze_dpt_backbone(self, num_blocks: int):
        try:
            backbone = getattr(self.dpt, "model", self.dpt).backbone
            pretrained = getattr(backbone, "pretrained", backbone)
            blocks = getattr(pretrained, "blocks", None)
        except Exception:
            return

        if blocks is not None:
            for idx, block in enumerate(blocks):
                if idx >= num_blocks:
                    break
                for p in block.parameters():
                    p.requires_grad_(False)

        if self.dpt_freeze_patch_embed and hasattr(pretrained, "patch_embed"):
            for p in pretrained.patch_embed.parameters():
                p.requires_grad_(False)
        if self.dpt_freeze_pos_embed:
            for name in ("pos_embed", "cls_token", "camera_token"):
                if hasattr(pretrained, name):
                    getattr(pretrained, name).requires_grad_(False)

    def _dpt_amp_dtype(self):
        if self.dpt_amp_dtype == "fp16":
            return torch.float16
        if self.dpt_amp_dtype == "bf16":
            return torch.bfloat16
        return torch.float32

    def _resize_for_dpt(self, imgs):
        B, N, C, H, W = imgs.shape
        if self.dpt_resize is not None:
            target_h, target_w = self.dpt_resize
        else:
            target_h = (H // self.dpt_patch_size) * self.dpt_patch_size
            target_w = (W // self.dpt_patch_size) * self.dpt_patch_size
            target_h = max(target_h, self.dpt_patch_size)
            target_w = max(target_w, self.dpt_patch_size)
        if target_h == H and target_w == W:
            return imgs, (H, W)
        imgs_flat = imgs.view(B * N, C, H, W)
        imgs_flat = F.interpolate(imgs_flat, size=(target_h, target_w), mode="bilinear", align_corners=False)
        return imgs_flat.view(B, N, C, target_h, target_w), (H, W)

    @staticmethod
    def _ensure_rgb(imgs):
        # DepthAnything expects 3-channel RGB input. Drop extra channels (e.g., alpha).
        if imgs.shape[2] > 3:
            return imgs[:, :, :3, ...]
        return imgs

    def _run_dpt(self, imgs):
        imgs = self._ensure_rgb(imgs)
        if next(self.dpt.parameters()).device != imgs.device:
            self.dpt.to(imgs.device)
        imgs_dpt, (H, W) = self._resize_for_dpt(imgs)
        device_type = "cuda" if imgs.device.type == "cuda" else "cpu"
        use_amp = self.dpt_use_amp and device_type == "cuda"
        with torch.no_grad():
            with torch.autocast(device_type=device_type, enabled=use_amp, dtype=self._dpt_amp_dtype()):
                raw = self.dpt(
                    imgs_dpt,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=self.dpt_feat_layers,
                    infer_gs=False,
                )
        return raw, (H, W)

    def _get_dpt_features(self, raw):
        aux = raw.get("aux", None)
        if aux is None:
            raise ValueError("DPT aux features are missing; check export_feat_layers.")
        feats = []
        for layer_idx in self.dpt_feat_layers:
            key = f"feat_layer_{layer_idx}"
            if key not in aux:
                raise ValueError(f"DPT aux missing key {key}.")
            feat = aux[key]
            # (B, N, Ht, Wt, C) -> (B, N, C, Ht, Wt)
            feat = feat.permute(0, 1, 4, 2, 3).contiguous()
            feats.append(feat)
        return feats

    def _build_layer_proj(self, feat_dims, device):
        self.layer_proj = nn.ModuleList(
            [nn.Conv2d(dim, self.camC, kernel_size=1).to(device) for dim in feat_dims]
        )

    def _fuse_dpt_features(self, feats):
        # feats: list of (B, N, C, H, W)
        if len(feats) == 1 and self.dpt_feat_fuse == "concat":
            feat = feats[0]
            B, N, C, H, W = feat.shape
            feat = feat.view(B * N, C, H, W)
            if self.feat_proj is None and C != self.camC:
                self.feat_proj = nn.Conv2d(C, self.camC, kernel_size=1).to(feat.device)
            if self.feat_proj is not None:
                feat = self.feat_proj(feat)
                C = feat.shape[1]
            return feat.view(B, N, C, H, W)

        B, N, _, H, W = feats[0].shape
        feat_dims = [f.shape[2] for f in feats]
        if self.dpt_feat_fuse == "concat":
            feat = torch.cat(feats, dim=2)
            feat = feat.view(B * N, feat.shape[2], H, W)
            if self.feat_proj is None:
                self.feat_proj = nn.Conv2d(feat.shape[1], self.camC, kernel_size=1).to(feat.device)
            feat = self.feat_proj(feat)
            return feat.view(B, N, self.camC, H, W)

        if self.layer_proj is None or len(self.layer_proj) != len(feats):
            self._build_layer_proj(feat_dims, feats[0].device)

        proj_feats = []
        for proj, f in zip(self.layer_proj, feats):
            f = f.view(B * N, f.shape[2], H, W)
            f = proj(f).view(B, N, self.camC, H, W)
            proj_feats.append(f)

        if self.dpt_feat_fuse == "sum":
            feat = torch.stack(proj_feats, dim=0).mean(dim=0)
        elif self.dpt_feat_fuse in ("weighted_sum", "pyramid"):
            weights = torch.softmax(self.layer_weights, dim=0)
            feat = sum(w * f for w, f in zip(weights, proj_feats))
            if self.dpt_feat_fuse == "pyramid":
                if self.fuse_conv is None:
                    self.fuse_conv = nn.Conv2d(self.camC, self.camC, kernel_size=3, padding=1).to(feat.device)
                feat = self.fuse_conv(feat.view(B * N, self.camC, H, W)).view(B, N, self.camC, H, W)
        else:
            raise ValueError(f"Unknown dpt_feat_fuse: {self.dpt_feat_fuse}")

        return feat

    def _build_depth_dist(self, depth, fH, fW):
        # depth: B x N x 1 x H x W
        B, N, _, H, W = depth.shape
        if H != fH or W != fW:
            depth = F.interpolate(depth.view(B * N, 1, H, W), size=(fH, fW), mode="bilinear", align_corners=False)
            depth = depth.view(B, N, 1, fH, fW)
        depth = depth.clamp(min=self.depth_min, max=self.depth_max)
        depth_flat = depth.view(B * N, fH, fW)
        depth_indices, mask = bin_depths(
            depth_flat,
            self.grid_conf["mode"],
            self.grid_conf["ddiscr"][0],
            self.grid_conf["ddiscr"][1],
            self.grid_conf["ddiscr"][2],
            target=self.training,
        )
        depth_dist = F.one_hot(depth_indices.long(), num_classes=self.grid_conf["ddiscr"][2]).permute(0, 3, 1, 2).float()
        if not self.training:
            mask = mask.unsqueeze(1)
            depth_dist = depth_dist * mask
        return depth_dist

    def _compute_depth_prior_cam(self, depth, depth_dist, fH, fW):
        if self.depth_prior_mode in (None, "none", "off"):
            return None
        depth_map = depth.squeeze(2)
        if depth_map.shape[-2:] != (fH, fW):
            depth_map = F.interpolate(depth_map.view(-1, 1, depth_map.shape[-2], depth_map.shape[-1]),
                                      size=(fH, fW), mode="bilinear", align_corners=False)
            depth_map = depth_map.view(depth_dist.shape[0], fH, fW)
        else:
            depth_map = depth_map.view(depth_dist.shape[0], fH, fW)
        if self.depth_prior_mode == "inverse_depth":
            conf = (self.depth_max - depth_map) / max(self.depth_max - self.depth_min, 1e-6)
            conf = conf.clamp(min=0.0, max=1.0)
        elif self.depth_prior_mode == "valid_mask":
            conf = (depth_map > 0).float()
        else:
            raise ValueError(f"Unknown depth_prior_mode: {self.depth_prior_mode}")
        conf = conf.view(depth_dist.shape[0], 1, fH, fW)
        depth_prior = depth_dist * conf
        return depth_prior.view(depth_dist.shape[0], self.D, fH, fW)

    def get_cam_feats(self, x):
        B, N, _, imH, imW = x.shape
        raw, (H, W) = self._run_dpt(x)
        depth = raw["depth"]  # (B, N, 1, h, w) or (B, N, h, w)
        if depth.dim() == 4:
            depth = depth.unsqueeze(2)
        elif depth.dim() != 5:
            raise ValueError(f"Unexpected DPT depth shape: {depth.shape}")
        feats = self._get_dpt_features(raw)

        # Align feature resolution to LSS frustum resolution
        fH, fW = imH // self.downsample, imW // self.downsample
        resized_feats = []
        for feat in feats:
            _, _, _, Ht, Wt = feat.shape
            if Ht != fH or Wt != fW:
                feat = feat.view(B * N, feat.shape[2], Ht, Wt)
                feat = F.interpolate(feat, size=(fH, fW), mode="bilinear", align_corners=False)
                feat = feat.view(B, N, feat.shape[1], fH, fW)
            resized_feats.append(feat)

        feat = self._fuse_dpt_features(resized_feats)

        depth_dist = self._build_depth_dist(depth, fH, fW)
        self._depth_prior_cam = None
        if self.depth_prior_mode not in (None, "none", "off"):
            depth_prior = self._compute_depth_prior_cam(depth, depth_dist, fH, fW)
            if depth_prior is not None:
                self._depth_prior_cam = depth_prior.view(B, N, self.D, fH, fW, 1)
        depth_dist = depth_dist.view(B * N, self.D, fH, fW)
        feat_flat = feat.view(B * N, self.camC, fH, fW)
        new_x = depth_dist.unsqueeze(1) * feat_flat.unsqueeze(2)
        new_x = new_x.view(B, N, self.camC, self.D, fH, fW)
        new_x = new_x.permute(0, 1, 3, 4, 5, 2)
        return new_x, None

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        x_img, depth_items = self.get_cam_feats(x)
        x = self.voxel_pooling(geom, x_img)
        if hasattr(self, "_depth_prior_cam") and self._depth_prior_cam is not None:
            depth_prior = self.voxel_pooling(geom, self._depth_prior_cam)
            self.depth_prior = depth_prior
        return x, depth_items
