import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Optional, Callable

# --- 核心构建模块：带预激活的1D残差块 ---
class ResidualBlock1D(nn.Module):
    """
    1D残差块，采用预激活 (LayerNorm -> Activation -> Conv) 结构。
    
    Attributes:
        features (int): 输出特征维度 (通道数)。
        kernel_size (int): 1D卷积核的大小。
    """
    features: int
    kernel_size: int
    activation: Callable
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: 输入张量，形状为 (batch, sequence_length, input_features)。
        
        Returns:
            输出张量，形状为 (batch, sequence_length, self.features)。
        """
        # 残差连接的输入
        residual = x
        
        # 预激活结构
        x = nn.LayerNorm()(x)
        x = self.activation(x)
        
        # 1D卷积层
        # padding='SAME' 保持序列长度不变，方便残差连接
        x = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,), # 1D卷积核大小需要是元组
            padding="SAME",
            kernel_init=nn.initializers.kaiming_normal(),
            bias_init=nn.initializers.constant(0.0)
        )(x)

        # 如果输入和输出的特征维度不同，需要对残差进行线性投影
        if residual.shape[-1] != self.features:
            residual = nn.Dense(
                features=self.features,
                kernel_init=nn.initializers.kaiming_normal(),
                bias_init=nn.initializers.constant(0.0)
            )(residual)
            
        return x + residual

class AttentionPooling(nn.Module):
    """
    通过自注意力机制实现的加权池化层。
    它会学习给序列中的每个时间步分配一个重要性权重。
    """
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: 输入张量，来自卷积层，形状为 (batch, sequence_length, features)。

        Returns:
            池化后的特征向量，形状为 (batch, features)。
        """
        # 1. 计算"重要性分数" (Attention Scores)
        # 使用一个简单的全连接层为每个时间步的特征向量计算一个分数
        # 输入: (B, S, F) -> 输出: (B, S, 1)
        scores = nn.Dense(features=1, name="Attention_Score_Dense")(x)
        
        # 2. 将分数转换为权重 (Softmax)
        # Softmax确保所有时间步的权重加起来等于1
        # scores 的形状是 (B, S, 1)，需要 reshape 成 (B, S) 来做 softmax
        weights = nn.softmax(scores.reshape(x.shape[0], x.shape[1]), axis=-1)
        
        # 3. 加权求和
        # weights的形状是(B, S)，需要扩展维度成(B, S, 1)才能与x广播相乘
        weighted_x = x * weights[..., None]
        
        # 沿时间步维度求和，得到最终的特征向量
        pooled_features = jnp.sum(weighted_x, axis=1)
        
        return pooled_features

class MultiHeadAttentionPooling(nn.Module):
    """
    使用多头注意力进行池化。
    
    Attributes:
        num_heads (int): 注意力头的数量。
    """
    num_heads: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: 输入张量 (batch, sequence_length, features)。

        Returns:
            池化后的特征向量 (batch, features)。
        """
        # Flax内置了高效的多头注意力实现
        # 我们用它来计算加权和，这在功能上等同于池化
        # 首先，我们需要一个“查询”向量(query)，它代表了我们“想问什么”
        # 在池化场景下，我们可以学习一个全局的“问题”，即[CLS] token
        cls_token = self.param('cls', nn.initializers.zeros, (1, 1, x.shape[-1]))
        cls_token = jnp.tile(cls_token, [x.shape[0], 1, 1]) # 复制到batch的每一项

        # MultiHeadDotProductAttention需要query, key, value
        # 在这里，query是我们的问题([CLS] token)，key和value都是K线序列x
        # 模型会学习用[CLS] token去"查询"K线序列x中最重要的信息
        x_pooled = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros
        )(inputs_q=cls_token, inputs_kv=x)
        
        # 输出形状是(B, 1, F)，我们需要去掉中间的维度
        return x_pooled.squeeze(axis=1)

# --- 主编码器 ---
class KLineEncoder(nn.Module):
    """
    用于K线数据的1D-CNN编码器。
    
    通过配置block_features和kernel_sizes可以轻松调整模型复杂度。
    
    Attributes:
        block_features (Sequence[int]): 每个残差块的输出特征维度列表。
                                       列表的长度决定了模型的深度。
        kernel_sizes (Sequence[int]): 每个残差块的卷积核大小列表。
    """
    block_features: Sequence[int]
    kernel_sizes: Sequence[int]
    activation: Callable
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Args:
            x: 输入K线数据，形状为 (batch_size, 30, 14)。
        
        Returns:
            提取的特征向量，形状为 (batch_size, last_block_feature)。
        """
        assert len(self.block_features) == len(self.kernel_sizes), "特征和卷积核列表长度必须一致"
        
        # 初始层：使用一个Dense层将14维特征投影到第一个块所需的维度
        # 这比用Conv(kernel_size=1)更直观
        x = nn.Dense(
            features=self.block_features[0],
            name="InitialProjection"
        )(x)
        
        # 堆叠残差块
        for i, (features, k_size) in enumerate(zip(self.block_features, self.kernel_sizes)):
            x = ResidualBlock1D(features=features, kernel_size=k_size, activation=self.activation, name=f"ResBlock1D_{i}")(x)
            
        # 全局平均池化 (Global Average Pooling)
        # 沿时间步维度（axis=1）取平均，得到一个固定大小的特征向量
        # 这比直接Flatten更常用，因为它对序列长度的变化不敏感
        # 将全局平均池化替换为注意力池化
        # x = jnp.mean(x, axis=1) # 这是原来的方法。有个问题：忽略时序信息，权重均等
        x = MultiHeadAttentionPooling(num_heads=8, name="MultiHeadAttentionPooling")(x) # 这是新方法
       
        
        return x