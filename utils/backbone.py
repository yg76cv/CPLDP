import torch
import torch.nn as nn


# 定义双卷积块，在 U - Net 的编码器和解码器中经常使用
class Double2s(nn.Module):
    def __init__(self, in_channels, out_channels):
        # 调用父类的构造函数
        super().__init__()
        # 定义一个顺序容器，包含两个卷积层、批量归一化层和 ReLU 激活函数
        self.conv = nn.Sequential(
            # 第一个卷积层，使用 3x3 卷积核，步长为 1，填充为 1，不使用偏置
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            # 批量归一化层，加速模型收敛
            nn.BatchNorm2d(out_channels),
            # ReLU 激活函数，引入非线性
            nn.ReLU(inplace=True),
            # 第二个卷积层，同样使用 3x3 卷积核，步长为 1，填充为 1，不使用偏置
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            # 批量归一化层
            nn.BatchNorm2d(out_channels),
            # ReLU 激活函数
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 前向传播，将输入 x 通过定义好的卷积层序列
        return self.conv(x)


# 定义 U - Net 模型
class UNet(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=3, features=[64, 128, 256, 512],condition_dim=512
    ):
        # 调用父类的构造函数
        super().__init__()
        # 定义一个模块列表，用于存储解码器中的上采样层和双卷积块
        self.ups = nn.ModuleList()
        # 定义一个模块列表，用于存储编码器中的双卷积块
        self.downs = nn.ModuleList()
        # 定义最大池化层，用于编码器中的下采样操作
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 编码器部分
        for feature in features:
            # 依次添加双卷积块到编码器模块列表中
            self.downs.append(Double2s(in_channels, feature))
            # 更新输入通道数为当前特征图的通道数
            in_channels = feature
        # 添加条件处理模块
        self.condition_dim = condition_dim
        self.condition_fc = nn.Sequential(
            nn.Linear(condition_dim, features[-1] * 4),
            nn.ReLU(inplace=True),
            nn.Linear(features[-1] * 4, features[-1] * 2)
        )
        # 修改瓶颈层，加入条件特征
        self.bottleneck = Double2s(features[-1], features[-1] * 2)
        self.bottleneck_condition = Double2s(features[-1] * 4, features[-1] * 2)  # 新瓶颈层

        # 解码器部分
        for feature in reversed(features):
            # 添加反卷积层，用于上采样操作
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=2, stride=2,
                )
            )
            # 添加双卷积块，用于特征融合和细化
            self.ups.append(Double2s(feature * 2, feature))

        # 最终卷积层，将解码器的输出映射到所需的通道数
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x, condition=None):
        # 用于存储编码器中每个阶段的特征图，以便在解码器中进行跳跃连接
        skip_connections = []

        # 编码器前向传播
        for down in self.downs:
            # 通过双卷积块提取特征
            x = down(x)
            # 保存当前特征图，用于后续的跳跃连接
            skip_connections.append(x)
            # 通过最大池化层进行下采样
            x = self.pool(x)

        # 瓶颈部分的前向传播
        x = self.bottleneck(x)
        if condition is not None:
            # 处理条件向量
            cond_feat = self.condition_fc(condition)
            cond_feat = cond_feat.view(cond_feat.size(0), -1, 1, 1)  # [B, C, 1, 1]
            cond_feat = cond_feat.expand(-1, -1, x.size(2), x.size(3))  # [B, C, H, W]

            # 拼接条件特征
            x = torch.cat([x, cond_feat], dim=1)
            x = self.bottleneck_condition(x)  # 通过新的瓶颈层
        # 反转跳跃连接列表，以便在解码器中按相反顺序使用
        skip_connections = skip_connections[::-1]

        # 解码器前向传播
        for idx in range(0, len(self.ups), 2):
            # 通过反卷积层进行上采样
            x = self.ups[idx](x)
            # 获取对应的跳跃连接特征图
            skip_connection = skip_connections[idx // 2]

            # 如果上采样后的特征图和跳跃连接特征图的尺寸不一致，进行插值操作
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(
                    x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True
                )

            # 将跳跃连接特征图和上采样后的特征图在通道维度上拼接
            concat_skip = torch.cat((skip_connection, x), dim=1)
            # 通过双卷积块进行特征融合和细化
            x = self.ups[idx + 1](concat_skip)

        # 通过最终卷积层得到输出
        return self.final_conv(x)


# 示例使用
if __name__ == "__main__":
    # 创建 U - Net 模型实例，输入通道数为 3，输出通道数为 3
    model = UNet(in_channels=3, out_channels=3)
    # 生成一个随机输入张量，模拟图像数据
    x = torch.randn(1, 3, 256, 256)
    # 进行前向传播，得到输出
    output = model(x)
    # 打印输出的形状
    print(output.shape)
