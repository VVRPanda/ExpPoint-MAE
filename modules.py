import numpy as np
import torch
import torch.nn as nn
import random
# TODO: Create custom DropPath to replace timm library
from timm.models.layers import DropPath

from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN

### Sampling, Grouping & Masking ###
def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

class Group(nn.Module):   # FPS + KNN

    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps to get the centers
        center = fps(xyz, self.num_group)
        # knn to get the neighborhood
        _, idx = self.knn(xyz, center) 
        #assert idx.size(1) == self.num_group
        #assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center

class Mask(nn.Module):

    def __init__(self, mask_ratio, mask_type='rand'):
        super().__init__()
        self.mask_ratio = mask_ratio 
        self.mask_type = mask_type

        if self.mask_type == 'rand':
            self.forward = self._mask_center_rand
        else:
            self.forward = self._mask_center_block

    def _mask_center_block(self, center, noaug=False):
        '''
            center: B x G x 3
            -----------------
            mask  : B x G (bool)
        '''
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool().to(center.device)
        # mask a continuous part
        mask_idx = []
        # processing each point cloud of the batch seperatelly 
        # NOTE / TODO: Could create the mask as a dataset transform?
        for points in center:
            # G x 3
            points = points.unsqueeze(0) # 1 x G x 3
            # selecting a random point 
            index = random.randint(0, points.size(1) - 1)
            # find the distance of this point to the other
            distance_matrix = torch.norm(points[:, index].reshape(1, 1, 3) - points, p=2,
                                         dim=-1) # 1x1x3 - 1xGx3 -> 1 x G
            # sort distances in ascending order --> return indexes
            idx = torch.argsort(distance_matrix, dim=-1, descending=False)[0] # G
            # find number of point to mask 
            ratio = self.mask_ratio
            mask_num = int(ratio * len(idx))
            # create the mask
            mask = torch.zeros(len(idx))
            # set masked values to 1
            mask[idx[:mask_num]] = 1
            mask_idx.append(mask.bool())

        bool_masked_pos = torch.stack(mask_idx).to(center.device) # B x G

        return bool_masked_pos


    def _mask_center_rand(self, center, noaug=False):
        '''
            center : B G 3
            --------------
            mask : B G (bool)
        '''
        B, G, _ = center.shape
        # skip the mask
        if noaug or self.mask_ratio == 0:
            return torch.zeros(center.shape[:2]).bool()

        num_mask = int(self.mask_ratio * G)

        overall_mask = np.zeros([B, G])
        for i in range(B):
            mask = np.hstack([
                np.zeros(G-num_mask),
                np.ones(num_mask),
            ])
            np.random.shuffle(mask)
            overall_mask[i, :] = mask
        
        overall_mask = torch.from_numpy(overall_mask).to(torch.bool)

        return overall_mask.to(center.device)

### Point Cloud Embedding Module ###
class PointEncoder(nn.Module):

    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1, bias=False), 
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1, bias=False), 
            nn.BatchNorm1d(512), 
            nn.ReLU(inplace=True), 
            nn.Conv1d(512, self.encoder_channel, 1)
        )

    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        B, G, N, _ = point_groups.shape
        point_groups = point_groups.reshape(B * G, N, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2, 1)) # B*G x 256 x N
        feature_global = torch.max(feature, dim=2, keepdim=True)[0] # B*G x 256 x 1
        # concating global features to each point features
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1) # B*G x 512 x N
        feature = self.second_conv(feature) # B*G x encoder_channel x N
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # B*G x encoder_channels
        return feature_global.reshape(B, G, self.encoder_channel)

### Transformer Modules ### (no relation with point clouds)
class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4)
        q, k , v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        
        mlp_hidden_dim = int(dim * mlp_ratio)

        # NOTE: Should test if this is better than dropout
        self.drop_path = DropPath(drop_path) if drop_path >0. else nn.Identity()

        # ATTENTION BLOCK
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        # MLP BLOCK
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Transformer(nn.Module):
        
    def __init__(self, embed_dim=768, depth=4, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., use_ViT_posemb=False):
        super().__init__()

        self.use_ViT_posemb = use_ViT_posemb
        
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, 
                drop=drop_rate, attn_drop=attn_drop_rate, 
                drop_path=drop_path_rate[i] if isinstance(drop_path_rate, list) else drop_path_rate
            )
        for i in range(depth)])

        # output norm 
        self.norm = nn.LayerNorm(embed_dim)

    def posemb_act(self, x, pos):
        for block in self.blocks:
            x = block(x + pos)

    def vit_act(self, x):
        for block in self.blocks:
            x = block(x)
            
    def forward(self, x, pos):
        # if self.use_ViT_posemb:
        # x = x + pos
        
        for block in self.blocks:
            # x = block(x)
            x = block(x + pos) # NOTE: add the positional embedding at every layer (why?)
        x = self.norm(x)
        return x

class TransformerWithEmbeddings(nn.Module):
    
    def __init__(self, embed_dim=768, depth=12, num_heads=12, feature_embed=True,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()

        if feature_embed:
            self.point_encoder = PointEncoder(embed_dim)
        else:
            self.point_encoder = nn.Identity()

        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim)
        )   

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = Transformer(
            embed_dim=embed_dim,
            depth=depth, 
            drop_path_rate=dpr, 
            num_heads=num_heads
        )


    def forward(self, neighs_of_feats, center, cls_token=None):

        # If feature_embed == False, then the network expects the neighborhood 
        # features as an input and uses an Identity layer as point encoder.
        # If feature_embed == True, then the network will
        # use the self.point_encoder to extract the feature embeddings. 
        x_vis = self.point_encoder(neighs_of_feats)

        # Extracting positional embeddings
        pos = self.pos_embedding(center)

        
        if cls_token is not None:
            b = x_vis.shape[0]
            cls_token = cls_token.expand(b, 1, -1)
            x_vis = torch.cat([x_vis, cls_token], dim=1)
            pos = torch.cat([pos, cls_token.new_zeros(cls_token.shape)], dim=1)

        # Activating the transformer layers
        x_vis = self.blocks(x_vis, pos)

        return x_vis

    
if __name__ == "__main__":
    # import time 
    # # TESTING MASKING MODULE
    # mask = Mask(0.5)
    # center = torch.rand(10, 64, 3)
    # t1 = time.time()
    # mask = mask(center)
    # t2 = time.time()
    # print(mask)
    # print(mask.shape)

    # print(f'Excecution time: {t2 - t1} sec')

    t = TransformerWithEmbeddings()

    print([(k, v, type(v)) for k, v in t.named_modules() if "adder" in k])