import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence, Optional, Literal

# 类型别名，增强可读性
Array = jnp.ndarray
Dtype = jnp.dtype

class ReparamLargeKernelConv(nn.Module):
    """
    结构重参数化大核卷积 (Structural Re-parameterization)。
    在训练时，使用一个大核卷积和一个小核卷积并行，以帮助优化。
    在推理时，可以等效合并为一个大核卷积以提升效率。
    此实现为训练版本。
    """
    features: int
    kernel_size: int
    small_kernel: int
    groups: int
    norm_type: Literal['batch', 'layer'] = 'layer'
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        # 大核卷积分支
        lkb_origin = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            strides=1,
            feature_group_count=self.groups,
            padding='SAME',
            use_bias=False,
            dtype=self.dtype,
            name='lkb_origin'
        )(x)
        if self.norm_type == 'batch':
            lkb_origin = nn.BatchNorm(
                use_running_average=not training,
                momentum=0.9,
                epsilon=1e-5,
                dtype=self.dtype,
                name='lkb_origin_bn'
            )(lkb_origin)
        else: # LayerNorm
             lkb_origin = nn.LayerNorm(dtype=self.dtype, name='lkb_origin_ln')(lkb_origin)


        # 小核卷积分支
        if self.small_kernel > 0:
            skb_origin = nn.Conv(
                features=self.features,
                kernel_size=(self.small_kernel,),
                strides=1,
                feature_group_count=self.groups,
                padding='SAME',
                use_bias=False,
                dtype=self.dtype,
                name='skb_origin'
            )(x)
            if self.norm_type == 'batch':
                skb_origin = nn.BatchNorm(
                    use_running_average=not training,
                    momentum=0.9,
                    epsilon=1e-5,
                    dtype=self.dtype,
                    name='skb_origin_bn'
                )(skb_origin)
            else: # LayerNorm
                skb_origin = nn.LayerNorm(dtype=self.dtype, name='skb_origin_ln')(skb_origin)
            
            return lkb_origin + skb_origin
        
        return lkb_origin

class ModernTCNBlock(nn.Module):
    """
    ModernTCN 的核心构建块。
    实现了时间、特征、变量三个维度的信息解耦处理。
    """
    d_model: int
    large_kernel_size: int
    small_kernel_size: int
    ffn_ratio: int = 2
    dropout_rate: float = 0.1
    norm_type: Literal['batch', 'layer'] = 'layer'
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        """
        Args:
            x (Array): 输入张量，形状为 (B, M, D, N) -> (Batch, n_vars, d_model, n_patches)
            training (bool): 是否为训练模式
        Returns:
            Array: 输出张量，形状与输入相同
        """
        B, M, D, N = x.shape
        residual = x
        
        # 1. Depth-wise Conv (时间维度混合)
        # 形状变换: (B, M, D, N) -> (B, N, M*D) 以便 Conv 处理
        x_dw = x.transpose((0, 3, 1, 2)).reshape(B, N, M * D)
        
        dw_groups = M * D # 每个单变量序列的每个特征通道独立卷积
        x_dw = ReparamLargeKernelConv(
            features=M * D,
            kernel_size=self.large_kernel_size,
            small_kernel=self.small_kernel_size,
            groups=dw_groups,
            norm_type=self.norm_type,
            dtype=self.dtype,
            name='dw_conv'
        )(x_dw, training=training)

        # 形状恢复并应用 Norm: (B, N, M*D) -> (B, M, D, N)
        x = x_dw.reshape(B, N, M, D).transpose((0, 2, 3, 1))

        # 论文中Norm作用于 D 维度
        x_norm = x.reshape(B * M, D, N)
        if self.norm_type == 'batch':
            # Flax BN作用于最后一个维度，所以需要转置
            x_norm = x_norm.transpose((0, 2, 1)) # (B*M, N, D)
            x_norm = nn.BatchNorm(
                use_running_average=not training,
                momentum=0.9,
                epsilon=1e-5,
                dtype=self.dtype
            )(x_norm).transpose((0, 2, 1)) # -> (B*M, D, N)
        else: # LayerNorm, 作用于最后一个维度
            x_norm = x_norm.transpose((0, 2, 1)) # (B*M, N, D)
            x_norm = nn.LayerNorm(dtype=self.dtype)(x_norm).transpose((0, 2, 1)) # -> (B*M, D, N)
        
        x = x_norm.reshape(B, M, D, N)

        # 2. ConvFFN1 (跨特征混合, 变量内)
        # 使用分组卷积实现，groups=M
        d_ffn = self.d_model * self.ffn_ratio
        x_ffn1 = x.transpose((0, 3, 1, 2)).reshape(B, N, M * D) # (B, N, M*D)
        
        ffn1_pw1 = nn.Conv(
            features=M * d_ffn,
            kernel_size=(1,),
            feature_group_count=M,
            dtype=self.dtype,
            name='ffn1_pw1'
        )(x_ffn1)
        ffn1_pw1 = nn.gelu(ffn1_pw1)
        ffn1_pw1 = nn.Dropout(self.dropout_rate)(ffn1_pw1, deterministic=not training)

        ffn1_pw2 = nn.Conv(
            features=M * D,
            kernel_size=(1,),
            feature_group_count=M,
            dtype=self.dtype,
            name='ffn1_pw2'
        )(ffn1_pw1)
        x_ffn1 = nn.Dropout(self.dropout_rate)(ffn1_pw2, deterministic=not training)
        
        # 3. ConvFFN2 (跨变量混合, 特征内)
        # 形状变换以进行跨变量卷积: (B, N, M, D) -> (B, N, D, M)
        x_ffn2 = x_ffn1.reshape(B, N, M, D).transpose((0, 1, 3, 2))
        x_ffn2 = x_ffn2.reshape(B, N, D * M)
        
        # 使用分组卷积实现，groups=D
        ffn2_pw1 = nn.Conv(
            features=D * d_ffn,
            kernel_size=(1,),
            feature_group_count=D,
            dtype=self.dtype,
            name='ffn2_pw1'
        )(x_ffn2)
        ffn2_pw1 = nn.gelu(ffn2_pw1)
        ffn2_pw1 = nn.Dropout(self.dropout_rate)(ffn2_pw1, deterministic=not training)
        
        ffn2_pw2 = nn.Conv(
            features=D * M,
            kernel_size=(1,),
            feature_group_count=D,
            dtype=self.dtype,
            name='ffn2_pw2'
        )(ffn2_pw1)
        x_ffn2 = nn.Dropout(self.dropout_rate)(ffn2_pw2, deterministic=not training)

        # 恢复形状并添加残差连接
        output = x_ffn2.reshape(B, N, D, M).transpose((0, 3, 2, 1)) # (B, M, D, N)
        
        return residual + output

class ModernTCNEncoder(nn.Module):
    """
    ModernTCN 模型，用作K线特征编码器。
    """
    n_vars: int
    patch_size: int
    patch_stride: int
    d_model: int
    num_blocks: int
    large_kernel_size: int
    small_kernel_size: int
    ffn_ratio: int = 2
    dropout_rate: float = 0.1
    revin: bool = False
    norm_type: Literal['batch', 'layer'] = 'layer'
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        """
        Args:
            x (Array): 输入K线数据, 形状为 (B, L, M) -> (Batch, seq_len, n_vars)
            training (bool): 是否为训练模式
        Returns:
            Array: 编码后的特征表示, 形状为 (B, M, D, N)
        """
        # RevIN: 可逆实例归一化，对时间序列稳定训练有益
        if self.revin:
            # 本质是先抹平价格的绝对位置以及波动率信息，输出时再应用回去。这样允许模型在一个稳定、标准的“内部世界”里进行学习和预测，最后再将结果映射回“现实世界”。使得模型可以专注于学习序列内部的形态和模式，而不会被样本本身的绝对值或波动大小干扰。
            # 但我的k线数据已经过LogReturn和StandardScaler处理变成了弱平稳的数据，作为Agent的特征提取器不应该用在输入seq内应用标准化，否则会丢失波动率性能
            raise NotImplementedError("RevIN is not implemented yet")
            # JAX 中通常在数据预处理阶段完成标准化，或自定义 Layer
            # 此处为简化，假设数据已做类似处理或实现一个 RevIN Layer
            # 简单实现InstanceNorm
            mean = x.mean(axis=1, keepdims=True)
            std = jnp.sqrt(x.var(axis=1, keepdims=True) + 1e-5)
            x = (x - mean) / std
        
        # 论文中的维度顺序是 (B, M, L), 与 PyTorch Conv1D 习惯一致
        # 我们调整输入以匹配：(B, L, M) -> (B, M, L)
        x = x.transpose((0, 2, 1))
        
        # 1. Patchify & Embedding (使用 Conv1D 实现)
        # (B, M, L) -> (B, M, 1, L) -> (B*M, 1, L)
        B, M, L = x.shape
        x_emb = x.reshape(B * M, 1, L)
        
        # 使用 strided Conv 实现 Patching 和 Embedding
        # 输出 (B*M, N, D)
        x_emb = nn.Conv(
            features=self.d_model,
            kernel_size=(self.patch_size,),
            strides=(self.patch_stride,),
            padding='VALID', # VALID 配合 stride 模拟 non-overlapping/overlapping patch
            dtype=self.dtype,
            name='embedding'
        )(x_emb)
        
        # 形状恢复: (B*M, N, D) -> (B, M, N, D) -> (B, M, D, N)
        _, N, D = x_emb.shape
        x_emb = x_emb.reshape(B, M, N, D).transpose(0, 1, 3, 2)
        
        # 2. Backbone: 堆叠 ModernTCN Block
        z = x_emb
        for i in range(self.num_blocks):
            z = ModernTCNBlock(
                d_model=self.d_model,
                large_kernel_size=self.large_kernel_size,
                small_kernel_size=self.small_kernel_size,
                ffn_ratio=self.ffn_ratio,
                dropout_rate=self.dropout_rate,
                norm_type=self.norm_type,
                dtype=self.dtype,
                name=f'moderntcn_block_{i}'
            )(z, training=training)
            
        return z # 输出 (B, M, D, N)

# --- 使用示例 ---
if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    
    # 模拟 K 线数据 (Batch, SeqLen, Vars)
    # 假设有 16 个样本，每个样本 128 个时间步长，5个变量 (O,H,L,C,V)
    batch_size = 16
    seq_len = 128
    n_vars = 5
    
    mock_kline_data = jnp.ones((batch_size, seq_len, n_vars))
    
    # 模型超参数 (参考论文和你的场景)
    encoder_params = {
        'n_vars': n_vars,
        'patch_size': 16,
        'patch_stride': 8,
        'd_model': 64,          # 特征维度
        'num_blocks': 3,        # Block 数量
        'large_kernel_size': 31,
        'small_kernel_size': 5,
        'ffn_ratio': 2,
        'dropout_rate': 0.1,
        'norm_type': 'layer',   # 默认使用 LayerNorm
    }

    # 初始化并应用模型
    encoder = ModernTCNEncoder(**encoder_params)
    
    # JAX/Flax 需要一个 `variables` 字典 (包含 `params` 和 `batch_stats`)
    variables = encoder.init({'params': key, 'dropout': key}, mock_kline_data, training=False)
    
    print("--- 初始化完成 ---")
    
    # 训练模式
    encoded_features_train, updated_states = encoder.apply(
        variables, 
        mock_kline_data, 
        training=True, 
        rngs={'dropout': key},
        mutable=['batch_stats'] # 如果使用BatchNorm，需要声明其为可变
    )
    
    # 推理模式
    encoded_features_eval = encoder.apply(
        variables,
        mock_kline_data,
        training=False
    )

    print("\n输入 K-Line 数据形状:", mock_kline_data.shape)
    print("编码后特征形状 (训练):", encoded_features_train.shape)
    print("编码后特征形状 (推理):", encoded_features_eval.shape)

    # 打印模型结构
    # print("\n模型结构:")
    # print(encoder.tabulate({'params': key, 'dropout': key}, mock_kline_data, training=False))