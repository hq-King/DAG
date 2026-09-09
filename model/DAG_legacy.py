import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from model.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from torchvision.ops import roi_align

from einops import rearrange
from model.dgcnn import DGCNN
from CLIP1.clip import cl as clip
from einops import rearrange
import timm
# import CLIP1.clip.cl as clip
from model.Uni3D.models import uni3d as modelss
from knn_cuda import KNN
import datetime
import os
import warnings
from PIL import Image
import cv2
import numpy as np
from model.odise.odise.modeling.meta_arch.ldm import LdmImplicitCaptionerExtractor
from model.odise.odise.modeling.backbone.feature_extractor import FeatureExtractorBackbone
from torchvision import transforms
import math





class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PreNorm_Atten(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, key_value):
        return self.fn(self.norm(x), self.norm(key_value))

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

# use Lemon's cross attention
class Cross_Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        self.inner_dim = dim_head *  heads
        self.dim_head = dim_head

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, self.inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, self.inner_dim*2, bias = False)

    def forward(self, query, key_value):

        B = query.size(0)
        q = self.to_q(query).view(B, -1, self.heads, self.dim_head).permute(0, 2, 1, 3)            #b n (h d)

        kv = self.to_kv(key_value).chunk(2, dim = -1)       
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), kv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.dropout(self.attend(dots))

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return out
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm_Atten(dim, Cross_Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, key_value):
        for attn, ff in self.layers:
            x = attn(x, key_value) + x
            x = ff(x) + x
        return x

##############
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # ReLU激活层
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # downsample层，用于调整维度
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
#######################
class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Conv2dBlock, self).__init__()
        # 卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # BatchNorm层
        self.bn = nn.BatchNorm2d(out_channels)
        # 激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out
    
class Embedder(nn.Module):
    def __init__(self):
        super(Embedder, self).__init__()
        
        # 使用卷积和池化来减少空间尺寸，并压缩通道数
        self.conv1 = nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1)  # 通道数从 2048 到 512  64
        self.conv2 = nn.Conv2d(1024,768, kernel_size=3, stride=2, padding=1)  # 通道数从 512 到 128    32
        self.conv3 = nn.Conv2d(768,768, kernel_size=3, stride=2, padding=1)  # 通道数从 128 到 64        16
        
        # 全连接层将最后的特征映射到 [b, 768]
        # self.fc = nn.Linear(64 * 16 * 16, 768)  # 64 * 16 * 16 是最后的展平后的特征维度

    def forward(self, x):
        # 逐层进行卷积和池化
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)#b 512,16,16
        
        # 展平
        x = x.view(x.size(0), x.size(1),-1)#b,512,256
        
        # 全连接层
        
        return x
    
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # 设计卷积模块（没有残差连接）
        self.conv1 = Conv2dBlock(1024, 512, kernel_size=3, stride=2, padding=1)
        self.conv2 = Conv2dBlock(512, 768, kernel_size=3, stride=2, padding=1)
        self.conv3 = Conv2dBlock(768, 768, kernel_size=3, stride=2, padding=1)
        self.conv4 = Conv2dBlock(768, 768, kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x
    
class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForwardNetwork, self).__init__()
        # 第一个线性变换
        self.linear1 = nn.Linear(d_model, d_ff)
        # 第二个线性变换
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # x 的形状为 [b, seq_len, d_model]
        b, seq_len, d_model = x.size()
        
        # 将输入展平为 [b * seq_len, d_model] 以应用线性变换
        x = x.view(b * seq_len, d_model)
        
        # 第一个线性变换后应用ReLU激活函数
        x = F.relu(self.linear1(x))
        
        # 第二个线性变换
        x = self.linear2(x)
        
        # 将输出恢复为原始形状 [b, seq_len, d_model]
        x = x.view(b, seq_len, d_model)
        
        return x
    
class AffordanceBlock(nn.Module):
    def __init__(self, input_dim_img, input_dim_text, hidden_dim, nhead, ff_dim):
        super(AffordanceBlock, self).__init__()

        # 图像输入的 Self-Attention 组件
        self.self_attn = nn.MultiheadAttention(embed_dim=input_dim_img, num_heads=nhead)
        
        # 前馈神经网络（含残差连接）
        self.ff = FeedForwardNetwork(768,2048)
        # 文本与图像特征的 Cross-Attention
        self.cross_attn = Transformer(dim = 512, depth = 1, heads = 8, mlp_dim = 512, dropout = 0.2, dim_head = 64)
        self.ln =  nn.Linear(768, 512)
        
        # 初始化一些需要的参数
        self.input_dim_img = input_dim_img
        self.input_dim_text = input_dim_text
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.prompt = nn.Parameter(torch.zeros(1, 1, 768))
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 层归一化
        self.layer_norm_img = nn.LayerNorm(input_dim_img)
        self.layer_norm_text = nn.LayerNorm(input_dim_text)
        
    def forward(self, img_features, text_features):
        # img_features: [batch_size, 768, 256]
        # text_features: [batch_size, 768]

        # Step 1: Self-Attention on image features
        # 需要调整图像特征的形状为 [seq_len, batch_size, feature_dim]
        img_features = img_features.permute(0, 2, 1)  # [b, 256, 768]
        
        # self-attention 过程
        img_features_self_attn, _ = self.self_attn(img_features, img_features, img_features)
        # img_features_self_attn = img_features_self_attn.permute(1, 2, 0)  # [b, 256, 768]
        # print(img_features_self_attn.shape)
        # Step 2: Feedforward on image features (with Residual Connection)
        img_features_ff = self.ff(img_features_self_attn)  # [b, 256, 768]
        # print(img_features_ff.shape)
        # Add residual connection
        img_features_ff = self.ln(img_features_self_attn + img_features_ff)#b,256,768

        # Step 3: Cross-Attention between image and text features
        # 文本特征需要调整为 [seq_len, batch_size, feature_dim] 形式
        # self.prompt = self.prompt.expand( img_features.shape[0],-1,-1)
        text_features = text_features.unsqueeze(0).permute(1,0,2)  # [256, 1, 768]
        prompt = self.ln(torch.cat((text_features, self.prompt.expand(img_features.shape[0], -1, -1)), dim=1)  )     # print(text_features.shape)
        # Cross-attention

        cross_attn_output = self.cross_attn( prompt,img_features_ff)
        # cross_attn_output = cross_attn_output.permute(1, 2, 0)  # [b, 768, 256]

       
        x_avg_pooled =self.avg_pool(cross_attn_output.transpose(1, 2))  # 需要将序列维度（第二维）转换为通道维度（第一维）
        x_avg_pooled = x_avg_pooled.squeeze(-1)  # 移除最后一个维度，大小为1
        # print(x_avg_pooled.shape)

        # Step 4: Combine and Output
        # # 进行归一化，最后通过加法跳跃连接
        # output = self.ln (self.layer_norm_img(img_features_ff + cross_attn_output))
        # attention_scores = torch.matmul(output, output.transpose(-1, -2)) / math.sqrt(768)
        # attention_weights = F.softmax(attention_scores, dim=-1)

        # # 使用注意力权重进行加权平均池化
        # output = torch.matmul(attention_weights,output).mean(dim=1).squeeze(1)  # 现在 
        # # print(output.shape)
        

        return x_avg_pooled
    
class Img_Encoder(nn.Module):
    def __init__(self):
        super(Img_Encoder, self).__init__()

        # self.model = models.resnet18(weights=None)
        # self.model.relu = nn.ReLU()

        ###################
        self.res = ConvNet()
        self.image_model = FeatureExtractorBackbone(
        feature_extractor=LdmImplicitCaptionerExtractor(
            encoder_block_indices=(5, 7),
            unet_block_indices=(2, 5, 8, 11),
            decoder_block_indices=(2, 5),
            steps=(0,),
            learnable_time_embed=True,
            num_timesteps=1,
            clip_model_name="ViT-L-14-336",
        ),
        out_features=["s2", "s3", "s4", "s5"],
        use_checkpoint=False,
        slide_training=True,
    ).cuda()
        checkpoint = torch.load("model/odise/feature_extractor.pth")

        self.image_model.load_state_dict(checkpoint, strict=False)     
        for name, param in self.image_model.named_parameters():            
            param.requires_grad = False
        self.embedder = Embedder()
        self.affordance_block = AffordanceBlock(768, 768, 1024, 8, 2048)
     

    def forward(self, afford_name,img):
#         transform = transforms.Compose([
#     transforms.Resize((512, 512)),  # 调整大小
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 使用与训练一致的均值和标准差
# ])

        # image_tensor = transform(img)
        # print(img.shape)
          # [1, C, H, W]
        with torch.no_grad():
            features,text_fea =  self.image_model(afford_name,img)
        feats = []
        for stage, feat in features.items():
            # print(f"Feature map from stage {stage}: {feat.shape}")
            if stage == "s2":
                f = feat
            if stage in ["s3","s4", "s5"]:
                # feat = torch.nn.functional.interpolate( feat, size=(128, 128), mode="bilinear", align_corners=False)
                feats.append(F.interpolate(feat, size=(128, 128), mode="bilinear", align_corners=False))
        feats.append(f)
        feats = torch.cat(feats, dim=1).cuda()#b,2048, 128 ,128
        # print(feats.shape)
        feats = self.embedder(feats)##b,768,256
        # print(feats.shape)
        # y = self.res(feats)
      
        # print(text_fea.shape )
        aff =  self.affordance_block(feats,text_fea)
        # exit(0)

        return aff

class Intention_Excavation(nn.Module):
    def __init__(self, input_dim, device):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        self.device = device
        # self.co_intention = Multi_Branch_Attention(dim = input_dim, heads = 12, dropout = 0.3, dim_head = 64)
        self.intention = Cross_Attention(dim = input_dim, heads = 8, dropout = 0.3, dim_head = 64)
        self.T_o = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.T_h = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.cosine = nn.CosineEmbeddingLoss()

    def forward(self ,F_i, F_o):
        ### TODO: no human mesh version
        B = F_i.size(0)

        F_to = torch.cat((self.T_o.expand(B,-1,-1), F_o), dim=1) # F_to: [16, 2049, 512]

        F_to_ = self.intention(F_to, F_i) # F_to_: [16, 2049, 512]

        T_o_, F_o_=  F_to_[:,0,:], F_to_[:,1:,:]

        return T_o_, F_o_
    
class Curvature_guided_Geometric_Correlation(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)

        self.f_m1 = Cross_Attention(dim = input_dim, heads = 8, dropout = 0.3, dim_head = 64)
        self.f_m2 = Cross_Attention(dim = input_dim, heads = 8, dropout = 0.3, dim_head = 64)

        self.fusion_hm = nn.Sequential(
            nn.Conv1d(input_dim*2, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(negative_slope=0.1),
            SwapAxes(),
        )

        self.fusion_obj = nn.Sequential(
            nn.Conv1d(input_dim*2, input_dim, 1),
            nn.BatchNorm1d(input_dim),
            nn.LeakyReLU(negative_slope=0.1),
            SwapAxes(),
        )

        self.affordance = Transformer(dim = input_dim, depth = 1, heads = 8, mlp_dim = 512, dropout = 0.3, dim_head = 64)
        self.contact = Transformer(dim = input_dim, depth = 1, heads = 8, mlp_dim = 512, dropout = 0.3, dim_head = 64)

    def forward(self, F_o_, T_o_):

        ### TODO: no human mesh version
        conditional_aff = T_o_.unsqueeze(dim=1)
        # print(F_o_.shape)
        # print(conditional_aff.shape)
        phi_a = self.affordance(conditional_aff,F_o_)
        # print(phi_a .shape)

        # exit(0)

        phi_a = self.affordance(F_o_, conditional_aff)

        return phi_a

# use Lemon's encoder
class Decoder(nn.Module):
    def __init__(self, feat_dim, device):
        super().__init__()
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        
        self.device = device
        self.aff_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim//6),
            SwapAxes(),
            nn.BatchNorm1d(feat_dim//6),
            SwapAxes(),
            nn.ReLU(),
            nn.Linear(feat_dim//6, 1)
        )

        self.contact_up_fine = nn.Linear(1723, 6890)
        self.sigmoid = nn.Sigmoid()

    def forward(self, phi_a):

        B = phi_a.size(0)
        affordance = self.aff_head(phi_a)                                  
        affordance = self.sigmoid(affordance)

        return affordance


class DITransformerAttentionModule(nn.Module):
    def __init__(self, in_channels=768, out_channels=512, num_heads=8):
        super(DITransformerAttentionModule, self).__init__()
        
        # Multihead Attention layer
        self.attn = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        
        # Linear layer for final output compression
        self.fc = nn.Linear(in_channels, out_channels)
        self.pool = nn.AdaptiveAvgPool1d(1)  # 将第二维（256）池化为1

        # Layer normalization
        self.layer_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x):
        batch_size, channels, height, width = x.shape
        
        # 展平空间维度 (height * width) 形成一个序列
        x = x.view(batch_size, channels, -1).permute(0, 2, 1)  # [batch_size, sequence_length, channels]
        
        # 应用多头自注意力
        attn_output, _ = self.attn(x, x, x)  # 输入、输出和注意力权重
        attn_output= attn_output.permute(0, 2, 1)
        attn_output = self.pool(attn_output)
        # print(attn_output.shape)
        # x = attn_output.permute(0, 2, 1).reshape(batch_size, channels, height, width)  # 恢复为原始尺寸
        
        # 将通道数压缩到512，展平除batch维度之外的所有维度
        # x = torch.flatten(x, 1)  # [batch_size, channels * height * width]
        
        # # 在此确保你使用展平后的大小作为fc的输入
        # # 比如：如果fc的输入维度是 `768 * 16 * 16`（196608），fc层的输入维度应该是196608
        # x = self.fc(x)  # [batch_size, out_channels]，这里的 out_channels 是 512
        
        # # 应用Layer Normalization
        # x = self.layer_norm(x)
        
        return attn_output.squeeze(-1)
    
#################
class DGCNN_Propagation(nn.Module):
    def __init__(self, k = 16):
        super().__init__()
        '''
        K has to be 16
        '''
        # print('using group version 2')
        self.k = k
        self.knn = KNN(k=k, transpose_mode=False)

        self.layer1 = nn.Sequential(nn.Conv2d(1536, 768, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 768),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

        self.layer2 = nn.Sequential(nn.Conv2d(1536, 768, kernel_size=1, bias=False),
                                   nn.GroupNorm(4, 768),
                                   nn.LeakyReLU(negative_slope=0.2)
                                   )

    @staticmethod
    def fps_downsample(coor, x, num_group):
        xyz = coor.transpose(1, 2).contiguous() # b, n, 3
        fps_idx = pointnet2_utils.furthest_point_sample(xyz, num_group)

        combined_x = torch.cat([coor, x], dim=1)

        new_combined_x = (
            pointnet2_utils.gather_operation(
                combined_x, fps_idx
            )
        )

        new_coor = new_combined_x[:, :3]
        new_x = new_combined_x[:, 3:]

        return new_coor, new_x

    def get_graph_feature(self, coor_q, x_q, coor_k, x_k):

        # coor: bs, 3, np, x: bs, c, np

        k = self.k
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            _, idx = self.knn(coor_k, coor_q)  # bs k np
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, coor, f, coor_q, f_q):
        """ coor, f : B 3 G ; B C G
            coor_q, f_q : B 3 N; B 3 N
        """
        # dgcnn upsample
        f_q = self.get_graph_feature(coor_q, f_q, coor, f)
        f_q = self.layer1(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        f_q = self.get_graph_feature(coor_q, f_q, coor_q, f_q)
        f_q = self.layer2(f_q)
        f_q = f_q.max(dim=-1, keepdim=False)[0]

        return f_q
    
class DAGLegacy(nn.Module):
    def __init__(self, img_model_path=None, pre_train = True, normal_channel=False, local_rank=None,
                N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, N_raw = 2048, num_affordance=18):
        class SwapAxes(nn.Module):
            def __init__(self):
                super().__init__()
            
            def forward(self, x):
                return x.transpose(1, 2)
        super().__init__()

        self.emb_dim = emb_dim
        self.N_p = N_p
        self.N_raw = N_raw
        self.proj_dim = proj_dim
        self.num_heads = num_heads
        self.local_rank = local_rank
        self.normal_channel = normal_channel
        self.num_affordance = num_affordance
        if self.normal_channel:
            self.additional_channel = 3
        else:
            self.additional_channel = 0

        self.img_encoder = Img_Encoder()

        device = torch.device("cuda:0")
        ####################
        model = modelss.create_uni3d()
        model.to(device)

        checkpoint = torch.load("ckpt/uni3d.pt", map_location=device)
        # logging.info('loaded checkpoint {}'.format(args.ckpt_path))
        sd = checkpoint['module']
        distributed = False
        if not distributed and next(iter(sd.items()))[0].startswith('module'):
            sd = {k[len('module.'):]: v for k, v in sd.items()}
        model.load_state_dict(sd)
        for name, param in model.named_parameters():            
            param.requires_grad = False
        self.uni3d = model
        # self.dfi = DITransformerAttentionModule(in_channels=768, out_channels=512)
        # self.trans_dim = 384



        # self.ln = nn.Linear(768, 512)
        

        # self.point_encoder = DGCNN(device, emb_dim=self.emb_dim)

        self.Intention = Intention_Excavation(self.emb_dim, device)
        self.Geometry_Correlation = Curvature_guided_Geometric_Correlation(self.emb_dim)
        self.LN = nn.Linear(768,512)
        self.decoder = Decoder(self.emb_dim, device)
        self.propagation_2 = PointNetFeaturePropagation(in_channel= 768+ 3, mlp = [768, 768])
        self.propagation_1= PointNetFeaturePropagation(in_channel= 768 + 3, mlp = [768 , 768])
        self.propagation_0 = PointNetFeaturePropagation(in_channel= 768 + 3+3, mlp = [768, 512])
        self.dgcnn_pro_1 = DGCNN_Propagation(k = 4)
        self.dgcnn_pro_2 = DGCNN_Propagation(k = 4)
        
    def forward(self,aff_name, img, xyz, sub_box, obj_box):

        '''
        img: [B, 3, H, W]
        xyz: [B, 3, 2048]
        sub_box: bounding box of the interactive subject
        obj_box: bounding box of the interactive object
        '''

        device = torch.device("cuda:0")
        B = img.size(0) # img: [16, 3, 224, 224], xyz: [16, 3, 2048]
        
        F_I = self.img_encoder(aff_name,img)      # F_I: [16, 49, 512]  



        rgb = torch.full_like(xyz, 0.4)
        xyz = xyz.permute(0,2,1)
        rgb = rgb.permute(0,2,1)
        fea =  torch.cat((xyz, rgb),dim=-1)
        
        h4,h8,h12,pts,center_level_0,center_level_1,center_level_2,center_level_3,cls_embedding= self.uni3d.encode_pc(fea)

        h4 = h4.permute(0,2,1)
        h8 = h8.permute(0,2,1)
        h12 = h12.permute(0,2,1)
        # cls_embedding =  self.LN(cls_embedding)

        f_level_1 = center_level_1
        f_level_2 = center_level_2
         # init the feature by 3nn propagation
        f_level_3 = h12
        f_level_2 = self.propagation_2(center_level_2, center_level_3, f_level_2, h8)#1024 feature
        f_level_1 = self.propagation_1(center_level_1, center_level_3, f_level_1, h4)#1536
        # bottom up
        f_level_2 = self.dgcnn_pro_2(center_level_3, f_level_3, center_level_2, f_level_2)
        f_level_1 = self.dgcnn_pro_1(center_level_2, f_level_2, center_level_1, f_level_1)
        f_level_0 =  self.propagation_0(center_level_0, center_level_1, fea.transpose(1,2), f_level_1)

        # print(up.shape)
        F_o = f_level_0 
        
        # F_I =  self.dfi(F_I)



        # F_I = self.ln(F_I)
        # print(F_o.shape)


        # print(F_I.shape)
        # T_o_, F_o_ = self.Intention(F_i, F_o.mT)       # T_o_[16, 512], F_o_[16, 2048, 512]
        phi_a = self.Geometry_Correlation(F_o.mT, F_I)  # phi_a[16, 2048, 512]

        affordance = self.decoder(phi_a)               # affordance[16, 2048, 1]

        return affordance

    def get_mask_feature(self, raw_img, img_feature, sub_box, obj_box, device):
        raw_size = raw_img.size(2)
        current_size = img_feature.size(2)
        B = img_feature.size(0)
        scale_factor = current_size / raw_size

        sub_box[:, :] = sub_box[:, :] * scale_factor
        obj_box[:, :] = obj_box[:, :] * scale_factor

        obj_mask = torch.zeros_like(img_feature) # obj_mask: [1, 512, 7, 7]
        obj_roi_box = []
        for i in range(B):
            obj_mask[i,:, int(obj_box[i][1]+0.5):int(obj_box[i][3]+0.5), int(obj_box[i][0]+0.5):int(obj_box[i][2]+0.5)] = 1
            roi_obj = [obj_box[i][0], obj_box[i][1], obj_box[i][2]+0.5, obj_box[i][3]]
            roi_obj.insert(0, i)
            obj_roi_box.append(roi_obj)
        obj_roi_box = torch.tensor(obj_roi_box).float().to(device) # obj_roi_box: [1, 5]

        sub_roi_box = []

        Scene_mask = obj_mask.clone()
        for i in range(B):
            Scene_mask[i,:, int(sub_box[i][1]+0.5):int(sub_box[i][3]+0.5), int(sub_box[i][0]+0.5):int(sub_box[i][2]+0.5)] = 1
            roi_sub = [sub_box[i][0], sub_box[i][1], sub_box[i][2], sub_box[i][3]]
            roi_sub.insert(0,i)
            sub_roi_box.append(roi_sub)
        Scene_mask = torch.abs(Scene_mask - 1)
        sub_roi_box = torch.tensor(sub_roi_box).float().to(device)
        obj_feature = roi_align(img_feature, obj_roi_box, output_size=(4,4), sampling_ratio=4)

        obj_feature = obj_feature.view(B, 512, -1)
        obj_feature = obj_feature.permute(0, 2, 1)

        return obj_feature

    def get_roi_box(self, batch_size):
        batch_box = []
        roi_box = [0., 0., 6., 6.]
        for i in range(batch_size):
            roi_box.insert(0, i)
            batch_box.append(roi_box)
            roi_box = roi_box[1:]

        batch_box = torch.tensor(batch_box).float()

        return batch_box

def get_DAGLegacy(img_model_path=None, pre_train = True, normal_channel=False, local_rank=None,
    N_p = 64, emb_dim = 512, proj_dim = 512, num_heads = 4, N_raw = 2048, num_affordance=17):
    
    model = DAGLegacy(img_model_path, pre_train, normal_channel, local_rank,
    N_p, emb_dim, proj_dim, num_heads, N_raw, num_affordance)
    return model