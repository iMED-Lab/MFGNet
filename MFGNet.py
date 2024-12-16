import torch.nn.functional as F
from rope import *


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)


class depthwise_separable_conv(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, padding=padding, groups=in_ch, bias=bias,
                                   stride=stride)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out


class RelativePositionBias(nn.Module):
    # input-independent relative position attention
    # As the number of parameters is smaller, so use 2D here
    # Borrowed some code from SwinTransformer: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
    def __init__(self, num_heads, h, w):
        super().__init__()
        self.num_heads = num_heads
        self.h = h
        self.w = w

        self.relative_position_bias_table = nn.Parameter(
            torch.randn((2 * h - 1) * (2 * w - 1), num_heads) * 0.02)

        coords_h = torch.arange(self.h)
        coords_w = torch.arange(self.w)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, h, w
        coords_flatten = torch.flatten(coords, 1)  # 2, hw

        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.h - 1
        relative_coords[:, :, 1] += self.w - 1
        relative_coords[:, :, 0] *= 2 * self.h - 1
        relative_position_index = relative_coords.sum(-1)  # hw, hw

        self.register_buffer("relative_position_index", relative_position_index)

    def forward(self, H, W):
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.h,
                                                                                                               self.w,
                                                                                                               self.h * self.w,
                                                                                                               -1)  # h, w, hw, nH
        relative_position_bias_expand_h = torch.repeat_interleave(relative_position_bias, H // self.h, dim=0)
        relative_position_bias_expanded = torch.repeat_interleave(relative_position_bias_expand_h, W // self.w,
                                                                  dim=1)  # HW, hw, nH

        relative_position_bias_expanded = relative_position_bias_expanded.view(H * W, self.h * self.w,
                                                                               self.num_heads).permute(2, 0,
                                                                                                       1).contiguous().unsqueeze(
            0)

        return relative_position_bias_expanded


class LinearAttention(nn.Module):

    def __init__(self, dim, heads=4, dim_head=64, attn_drop=0., proj_drop=0., reduce_size=8, projection='interp',
                 rel_pos=True):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** (-0.5)
        self.dim_head = dim_head
        self.reduce_size = reduce_size
        self.projection = projection
        self.rel_pos = rel_pos

        # depthwise conv is slightly better than conv1x1
        # self.to_qkv = nn.Conv2d(dim, self.inner_dim*3, kernel_size=1, stride=1, padding=0, bias=True)
        # self.to_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, stride=1, padding=0, bias=True)

        self.to_qkv = depthwise_separable_conv(dim, self.inner_dim * 3)
        self.to_out = depthwise_separable_conv(self.inner_dim, dim)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.rel_pos:
            # 2D input-independent relative position encoding is a little bit better than
            # 1D input-denpendent counterpart
            self.relative_position_encoding = RelativePositionBias(heads, reduce_size, reduce_size)
            # self.relative_position_encoding = RelativePositionEmbedding(dim_head, reduce_size)

    def forward(self, x):

        B, C, H, W = x.shape

        # B, inner_dim, H, W
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        if self.projection == 'interp' and H != self.reduce_size:
            k, v = map(lambda t: F.interpolate(t, size=self.reduce_size, mode='bilinear', align_corners=True), (k, v))

        elif self.projection == 'maxpool' and H != self.reduce_size:
            k, v = map(lambda t: F.adaptive_max_pool2d(t, output_size=self.reduce_size), (k, v))

        q = rearrange(q, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head, heads=self.heads,
                      h=H, w=W)
        k, v = map(lambda t: rearrange(t, 'b (dim_head heads) h w -> b heads (h w) dim_head', dim_head=self.dim_head,
                                       heads=self.heads, h=self.reduce_size, w=self.reduce_size), (k, v))

        q_k_attn = torch.einsum('bhid,bhjd->bhij', q, k)

        if self.rel_pos:
            relative_position_bias = self.relative_position_encoding(H, W)
            q_k_attn += relative_position_bias
            # rel_attn_h, rel_attn_w = self.relative_position_encoding(q, self.heads, H, W, self.dim_head)
            # q_k_attn = q_k_attn + rel_attn_h + rel_attn_w

        q_k_attn *= self.scale
        q_k_attn = F.softmax(q_k_attn, dim=-1)
        q_k_attn = self.attn_drop(q_k_attn)

        out = torch.einsum('bhij,bhjd->bhid', q_k_attn, v)
        out = rearrange(out, 'b heads (h w) dim_head -> b (dim_head heads) h w', h=H, w=W, dim_head=self.dim_head,
                        heads=self.heads)

        out = self.to_out(out)
        out = self.proj_drop(out)

        return out


class BasicTransBlock(nn.Module):

    def __init__(self, in_ch, heads, dim_head, attn_drop=0., proj_drop=0., reduce_size=8, projection='interp',
                 rel_pos=True):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)

        self.attn = LinearAttention(in_ch, heads=heads, dim_head=in_ch // heads, attn_drop=attn_drop,
                                    proj_drop=proj_drop, reduce_size=reduce_size, projection=projection,
                                    rel_pos=rel_pos)

        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.mlp = nn.Conv2d(in_ch, in_ch, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.bn1(x)
        out = self.attn(out)

        out = out + x
        residue = out

        out = self.bn2(out)
        out = self.relu(out)
        out = self.mlp(out)

        out += residue

        return out


class ConvNormAct(nn.Module):
    """
    Layer grouping a convolution, normalization and activation funtion
    normalization includes BN and IN
    """

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=0,
                 groups=1, dilation=1, bias=False, norm=nn.InstanceNorm2d, act=nn.GELU, preact=False):

        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias
        )
        if preact:
            self.norm = norm(in_ch) if norm else nn.Identity()
        else:
            self.norm = norm(out_ch) if norm else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.preact = preact

    def forward(self, x):

        if self.preact:
            out = self.conv(self.act(self.norm(x)))
        else:
            out = self.act(self.norm(self.conv(x)))

        return out


class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, norm=nn.InstanceNorm2d, act=nn.GELU, preact=False):
        super().__init__()
        assert norm in [nn.BatchNorm2d, nn.InstanceNorm2d, True, False]
        assert act in [nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, True, False]

        self.conv1 = ConvNormAct(in_ch, out_ch, 3, stride=stride, padding=1, norm=norm, act=act, preact=preact)
        self.conv2 = ConvNormAct(out_ch, out_ch, 3, stride=1, padding=1, norm=norm, act=act, preact=preact)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = ConvNormAct(in_ch, out_ch, 1, norm=norm, act=act, preact=preact)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        out += self.shortcut(residual)

        return out


class upConv_Ps(nn.Module):
    def __init__(self, in_ch, out_ch, num_block=1, scale_factor=None, block=BasicBlock):
        super().__init__()
        self.scale_factor = scale_factor

        self.conv_ch = nn.Conv2d(in_ch, out_ch, kernel_size=1)

        block_list = []
        block_list.append(block(2 * out_ch, out_ch))

        for i in range(num_block - 1):
            block_list.append(block(out_ch, out_ch))

        self.conv = nn.Sequential(*block_list)

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, scale_factor=self.scale_factor, mode='bilinear', align_corners=True)
        x1 = self.conv_ch(x1)

        out = torch.cat([x2, x1], dim=1)
        out = self.conv(out)

        return out


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(in_ch, out_ch, 3, padding=1 * dirate, dilation=1 * dirate)
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


## upsample tensor 'src' to have the same spatial size with tensor 'tar'
def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode='bilinear', align_corners=True)

    return src


class RSU_simple(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU_simple, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)
        # hx4 = hx4 * attention + hx4

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class RSU(nn.Module):  # UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x, attention):
        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx4 = hx4 * attention + hx4

        hx3d = self.rebnconv6d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# stem操作
class Stem(nn.Module):

    def __init__(self, in_chans=1, out_chans=32, stride=1):
        super().__init__()
        self.conv1 = conv3x3(in_chans, out_chans, stride)
        self.bn1 = nn.BatchNorm2d(out_chans)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(out_chans, out_chans)
        self.bn2 = nn.BatchNorm2d(out_chans)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_chans != out_chans:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chans, out_chans, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chans),
                self.relu
            )

    def forward(self, x):
        residue = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = out + self.shortcut(residue)

        return out


class DFG_module(nn.Module):
    def __init__(self, in_ch, heads):
        super().__init__()

        self.conv_block = RSU(in_ch=in_ch, mid_ch=in_ch // 2, out_ch=in_ch)
        self.transformer = BasicTransBlock(in_ch=in_ch, heads=heads, dim_head=in_ch // heads)

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, in_ch // 4, kernel_size=1, stride=1),
            nn.BatchNorm2d(in_ch // 4),
            nn.GELU(),
            nn.Conv2d(in_ch // 4, in_ch, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 4, kernel_size=7, stride=4, padding=3, groups=in_ch // 4),
            nn.InstanceNorm2d(in_ch // 4),
            nn.GELU(),
            nn.Conv2d(in_ch // 4, in_ch // 2, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.conv1_1 = nn.Conv2d(in_ch * 2, in_ch, kernel_size=1, stride=1)
        self.bn = nn.InstanceNorm2d(in_ch)

        self.ffn = Mlp(in_ch, in_ch * 4, in_ch, drop=0.0)

    def forward(self, x):
        x_transformer = self.transformer(x)
        x_spatial_attention = self.spatial_attention(x_transformer)
        x_conv = self.conv_block(x, x_spatial_attention)
        x_channel_attention = self.channel_attention(x_conv)
        x_transformer = x_transformer * x_channel_attention

        x = self.conv1_1(torch.cat([x_transformer, x_conv], dim=1))

        x = self.ffn(x)

        return x


class LSKAblock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=3, groups=dim)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 1, kernel_size=1)
        self.conv = nn.Conv2d(dim // 2, dim, 1)

    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(x)

        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)

        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        # attn = self.conv(attn)
        x = x * sig + x

        return x


class choroidal_encoder_block(nn.Module):
    def __init__(self, in_dim, out_dim, heads, depth):
        super().__init__()
        self.max_pool = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1)
        )

        self.layers = nn.ModuleList(
            [
                DFG_module(out_dim, heads)
                for i in range(depth)
            ]
        )

    def forward(self, x):
        x = self.max_pool(x)

        for layer in self.layers:
            feature_out = layer(x)

        return feature_out


class choroidal_bottleneck_block(nn.Module):
    def __init__(self, in_dim, out_dim, depth, stride=1):
        super().__init__()
        self.conv_1 = nn.Sequential(
            conv3x3(in_dim, out_dim),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        self.layers = nn.ModuleList(
            [
                LSKAblock(out_dim)
                for i in range(depth)
            ]
        )

        self.conv_2 = nn.Sequential(
            conv3x3(out_dim, out_dim),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        residue = x

        x = self.conv_1(x)

        for layer in self.layers:
            feature_out = layer(x)
        feature_out = feature_out + x
        feature_out = self.conv_2(feature_out)

        feature_out = feature_out + self.shortcut(residue)

        return feature_out


class choroidal_encoder(nn.Module):
    def __init__(self, input_channels=1, dims=[32, 64, 128, 256, 512], pool_size=3, num_heads=[8, 4, 4, 2],
                 depths=[1, 2, 2, 2, 2]):
        super().__init__()

        self.stem = choroidal_bottleneck_block(in_dim=input_channels, out_dim=dims[0], depth=depths[0])
        self.encoder0 = choroidal_encoder_block(in_dim=dims[0], out_dim=dims[1], heads=num_heads[0], depth=depths[1])
        self.encoder1 = choroidal_encoder_block(in_dim=dims[1], out_dim=dims[2], heads=num_heads[1], depth=depths[2])
        self.encoder2 = choroidal_encoder_block(in_dim=dims[2], out_dim=dims[3], heads=num_heads[2], depth=depths[3])
        self.encoder3 = choroidal_encoder_block(in_dim=dims[3], out_dim=dims[4], heads=num_heads[3], depth=depths[4])

    def forward(self, x):
        x_stem = self.stem(x)
        x1 = self.encoder0(x_stem)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)

        return x1, x2, x3, x4, x_stem


class choroidal_decoder(nn.Module):
    def __init__(self, dims=[32, 64, 128, 256, 512], num_classes=10, deep_supervision=True):
        super().__init__()

        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.up_layer0 = upConv_Ps(in_ch=dims[3] * 2, out_ch=dims[3], scale_factor=2)
        self.up_layer1 = upConv_Ps(in_ch=dims[3], out_ch=dims[2], scale_factor=2)
        self.up_layer2 = upConv_Ps(in_ch=dims[2], out_ch=dims[1], scale_factor=2)
        self.up_layer3 = upConv_Ps(in_ch=dims[1], out_ch=dims[0], scale_factor=2)

        self.out_1 = nn.Conv2d(dims[3], num_classes, kernel_size=1)
        self.out_2 = nn.Conv2d(dims[2], num_classes, kernel_size=1)
        self.out_3 = nn.Conv2d(dims[1], num_classes, kernel_size=1)
        self.out_4 = nn.Conv2d(dims[0], num_classes, kernel_size=1)

        self.final_conv = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def forward(self, x1, x2, x3, x4, x_skip):
        seg_out = []

        skip1 = self.up_layer0(x4, x3)  # 12 768 16 16
        out1 = self.out_1(skip1)

        skip2 = self.up_layer1(skip1, x2)  # 12 384 32 32
        out2 = self.out_2(skip2)

        skip3 = self.up_layer2(skip2, x1)  # 12 192 64 64
        out3 = self.out_3(skip3)

        skip4 = self.up_layer3(skip3, x_skip)  # 12 96 128 128
        out = self.final_conv(skip4)

        seg_out.append(out1)
        seg_out.append(out2)
        seg_out.append(out3)
        seg_out.append(out)

        seg_outputs = seg_out[::-1]

        if self.deep_supervision:
            return seg_outputs
        else:
            return seg_outputs[0]


class MFGNet(nn.Module):
    def __init__(self, input_channels=1, dims=[32, 64, 128, 256, 512], num_classes=2, depths=[1, 3, 4, 6, 3],
                 deep_supervision=True):
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision

        self.encoder = choroidal_encoder(input_channels=input_channels, dims=dims, depths=depths)

        self.decoder = choroidal_decoder(dims=dims, num_classes=num_classes, deep_supervision=deep_supervision)

    def forward(self, x):
        x1, x2, x3, x4, x_skip = self.encoder(x)

        return self.decoder(x1, x2, x3, x4, x_skip)
