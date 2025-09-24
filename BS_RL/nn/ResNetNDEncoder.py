import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze

from typing import Sequence, Type, Callable
from dataclasses import dataclass


@dataclass
class ResNetConfig:
    """
    用于配置 N 维 ResNet 模型的 Dataclass。

    Attributes:
        stage_sizes: 一个序列，定义了每个 stage 中残差块的数量。
                     例如 ResNet-18 的是 [2, 2, 2, 2]。
        block_cls: 要使用的残差块的类 (ResidualBlock 或 BottleneckBlock)。
        dropout_rate: 在残差块内部应用的 Dropout 比率 (0.0 表示不使用)。
        enable_initial_max_pool: 是否在 Stem 后立即进行一次 3x...x3 max-pool，下采样
    """
    stage_sizes: Sequence[int]
    block_cls: Type[nn.Module]
    dropout_rate: float = 0.1
    enable_initial_max_pool: bool = False
    stem_features: int = 64
    """标准ResNet的Stem层输出特征数是64"""


class ResidualBlock(nn.Module):
    """基础残差块 (用于 ResNet-18/34) 的 N 维版本。"""
    features: int
    activation: Callable
    strides: int = 1
    dropout_rate: float = 0.0
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        ndim = x.ndim - 2
        strides_tuple = (self.strides,) * ndim
        
        # 残差连接的捷径，保持原始输入
        shortcut = x
        
        # 预激活: LayerNorm -> GELU
        y = nn.LayerNorm()(x)
        y = self.activation(y)

        # 如果需要下采样或改变特征维度，则需要对捷径进行投影
        if x.shape[-1] != self.features or self.strides != 1:
             shortcut = nn.Conv(
                features=self.features, 
                kernel_size=(1,) * ndim,
                strides=strides_tuple, 
                name='shortcut_conv'
            )(x)

        # 第一个卷积层
        y = nn.Conv(
            features=self.features,
            kernel_size=(3,) * ndim,
            strides=strides_tuple,
            padding='SAME', # 使用 SAME padding 来处理 stride
            name='conv1'
        )(y)

        # 第二个卷积层前的预激活
        y = nn.LayerNorm()(y)
        y = self.activation(y)
        
        y = nn.Conv(
            features=self.features,
            kernel_size=(3,) * ndim,
            padding='SAME',
            name='conv2'
        )(y)
        
        # 在 block 末尾加 dropout（如果启用）
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(y)
            
        # 加上捷径后再做一次激活（符合 ResNet-v2 建议）
        out = y + shortcut
        return self.activation(out)

class BottleneckBlock(nn.Module):
    """瓶颈残差块 (用于 ResNet-50+) 的 N 维版本。"""
    features: int
    activation: Callable
    strides: int = 1
    dropout_rate: float = 0.0
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        ndim = x.ndim - 2
        strides_tuple = (self.strides,) * ndim
        
        shortcut = x
        
        # 预激活: LayerNorm -> GELU
        y = nn.LayerNorm()(x)
        y = self.activation(y)
        
        # 瓶颈块的输出特征数是输入的4倍
        bottleneck_features = self.features * 4

        # 如果需要下采样或改变特征维度，对捷径进行投影
        if x.shape[-1] != bottleneck_features or self.strides != 1:
             shortcut = nn.Conv(
                features=bottleneck_features, 
                kernel_size=(1,) * ndim,
                strides=strides_tuple,
                name='shortcut_conv'
            )(x)

        # 1x...x1 卷积: 降维
        y = nn.Conv(
            features=self.features,
            kernel_size=(1,) * ndim,
            name='conv1'
        )(y)

        # 3x...x3 卷积: 特征提取 (可能带下采样)
        y = nn.LayerNorm()(y)
        y = self.activation(y)
        
        y = nn.Conv(
            features=self.features,
            kernel_size=(3,) * ndim,
            strides=strides_tuple,
            padding='SAME',
            name='conv2'
        )(y)
        
        # 1x...x1 卷积: 升维
        y = nn.LayerNorm()(y)
        y = self.activation(y)
        y = nn.Conv(
            features=bottleneck_features,
            kernel_size=(1,) * ndim,
            name='conv3'
        )(y)
        
        # 在 block 末尾加 dropout（如果启用）
        if self.dropout_rate > 0.0:
            y = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(y)
        
        # 加上捷径后再做一次激活（符合 ResNet-v2 建议）
        out = y + shortcut
        return self.activation(out)


class ResNetNDEncoder(nn.Module):
    """
    N 维 ResNet编码器，用于将时序或图像数据编码为特征向量。
    """
    config: ResNetConfig
    activation: Callable = nn.gelu
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool) -> jnp.ndarray:
        ndim = x.ndim - 2
        
        # 初始卷积层 (Stem)
        # 类似于图像的 7x7 卷积，这里用 7x...x7，并用步长2做第一次下采样
        x = nn.Conv(
            features=self.config.stem_features, 
            kernel_size=(7,) * ndim,
            strides=(2,) * ndim,
            padding='SAME', 
            name='conv_initial'
        )(x)
        
        # 初始池化层（可选），进一步下采样
        if self.config.enable_initial_max_pool:
            x = nn.max_pool(x, window_shape=(3,) * ndim, strides=(2,) * ndim, padding='SAME')
        
        # ResNet Stages
        # 循环创建每个阶段的残差块
        for i, block_count in enumerate(self.config.stage_sizes):
            # 每个 stage 的特征数翻倍 (64 -> 128 -> 256 -> 512)
            stage_features = self.config.stem_features * (2**i)
            for j in range(block_count):
                # 除了第一个 stage，每个 stage 的第一个 block 进行下采样 (stride=2)
                block_strides = 2 if i > 0 and j == 0 else 1
                
                x = self.config.block_cls(
                    features=stage_features, 
                    strides=block_strides,
                    dropout_rate=self.config.dropout_rate,
                    activation=self.activation
                )(x, training=training)

        # In pre-activation mode, a final normalization and activation is applied before pooling.
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        
        # 全局平均池化
        spatial_axes = tuple(range(1, ndim + 1))
        x = jnp.mean(x, axis=spatial_axes)
        
        return x

# --- 工厂函数：用于方便地创建不同深度的 ResNet 模型 ---
def ResNet8(dropout_rate: float = 0.0, **kwargs) -> ResNetNDEncoder:
    """创建 N 维 ResNet-8 模型（轻量版）。"""
    config = ResNetConfig(
        stage_sizes=[1, 1, 1], 
        block_cls=ResidualBlock,
        dropout_rate=dropout_rate
    )
    return ResNetNDEncoder(config=config, **kwargs)

def ResNet18(dropout_rate: float = 0.0, **kwargs) -> ResNetNDEncoder:
    """创建 N 维 ResNet-18 模型。"""
    config = ResNetConfig(
        stage_sizes=[2, 2, 2, 2], 
        block_cls=ResidualBlock,
        dropout_rate=dropout_rate
    )
    return ResNetNDEncoder(config=config, **kwargs)

def ResNet34(dropout_rate: float = 0.0, **kwargs) -> ResNetNDEncoder:
    """创建 N 维 ResNet-34 模型。"""
    config = ResNetConfig(
        stage_sizes=[3, 4, 6, 3], 
        block_cls=ResidualBlock,
        dropout_rate=dropout_rate
    )
    return ResNetNDEncoder(config=config, **kwargs)

def ResNet50(dropout_rate: float = 0.0, **kwargs) -> ResNetNDEncoder:
    """创建 N 维 ResNet-50 模型。"""
    config = ResNetConfig(
        stage_sizes=[3, 4, 6, 3], 
        block_cls=BottleneckBlock,
        dropout_rate=dropout_rate
    )
    return ResNetNDEncoder(config=config, **kwargs)




