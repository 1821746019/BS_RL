import jax
import jax.numpy as jnp
from jax import lax
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

class ModernTCNStage(nn.Module):
    """一个 ModernTCN Stage，包含多个 ModernTCNBlock"""
    num_blocks: int
    d_model: int
    large_kernel_size: int
    small_kernel_size: int
    ffn_ratio: int
    dropout_rate: float
    norm_type: Literal['batch', 'layer']
    dtype: Dtype

    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        for i in range(self.num_blocks):
            x = ModernTCNBlock(
                d_model=self.d_model,
                large_kernel_size=self.large_kernel_size,
                small_kernel_size=self.small_kernel_size,
                ffn_ratio=self.ffn_ratio,
                dropout_rate=self.dropout_rate,
                norm_type=self.norm_type,
                dtype=self.dtype,
                name=f'moderntcn_block_{i}'
            )(x, training=training)
        return x

class FlattenHead(nn.Module):
    """
    将 (B, M, D, N) 的输出展平并映射到 target_window 长度的预测头。
    """
    n_vars: int
    head_nf: int
    target_window: int
    individual: bool = False
    head_dropout: float = 0.1
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        # x: [B, M, D, N]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = x[:, i, :, :].transpose(0, 2, 1).reshape(x.shape[0], -1) # (B, D, N) -> (B, N, D) -> (B, N*D)
                z = nn.Dense(features=self.target_window, dtype=self.dtype, name=f'dense_{i}')(z)
                z = nn.Dropout(self.head_dropout)(z, deterministic=not training)
                x_out.append(z)
            x = jnp.stack(x_out, axis=1) # (B, M, target_window)
        else:
            # (B, M, D, N) -> (B, M, N, D) -> (B, M, N*D)
            x = x.transpose(0, 1, 3, 2).reshape(x.shape[0], self.n_vars, -1)
            x = nn.Dense(features=self.target_window, dtype=self.dtype, name='dense')(x)
            x = nn.Dropout(self.head_dropout)(x, deterministic=not training)
        
        return x.transpose(0, 2, 1) # (B, target_window, n_vars)

class ModernTCNEncoder(nn.Module):
    """
    ModernTCN 编码器，用于从时间序列中提取丰富的特征表示。
    输出形状为 (B, M, D, N)。
    """
    patch_size: int
    patch_stride: int
    dims: Sequence[int]
    num_blocks: Sequence[int]
    large_kernel_sizes: Sequence[int]
    small_kernel_sizes: Sequence[int]
    downsample_ratio: int
    n_vars: int = -1 # 在flax，shape能自动推断而无需该字段
    ffn_ratio: int = 2
    dropout_rate: float = 0.1
    revin: bool = False
    norm_type: Literal['batch', 'layer'] = 'layer'
    dtype: Dtype = jnp.float32
    post_proc: Literal['none', 'max_pool', 'avg_pool', 'global_max_pool', 'global_avg_pool'] = 'global_avg_pool'
    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        """
        Args:
            x (Array): 输入K线数据, 形状为 (B, L, M) -> (Batch, seq_len, n_vars)
            training (bool): 是否为训练模式
        Returns:
            Array: 编码后的特征, 形状为 (B, M, D, N)
        """
        # RevIN: 可逆实例归一化，对时间序列稳定训练有益
        if self.revin:
            # 对于价格预测有用：本质是先抹平价格的绝对位置以及波动率信息，输出时再应用回去。这样允许模型在一个稳定、标准的“内部世界”里进行学习和预测，最后再将结果映射回“现实世界”。使得模型可以专注于学习序列内部的形态和模式，而不会被样本本身的绝对值或波动大小干扰。
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
        
        B, M, L = x.shape
        
        # (B, M, L) -> (B*M, L, 1)
        x_emb = x.reshape(B * M, L, 1)

        # 1. Stem and Downsampling Backbone
        num_stages = len(self.dims)
        
        for i in range(num_stages):
            if i == 0: # Stem layer
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    x_emb = jnp.pad(x_emb, ((0, 0), (0, pad_len), (0, 0)), mode='edge')
                
                x_emb = nn.Conv(
                    features=self.dims[i],
                    kernel_size=(self.patch_size,),
                    strides=(self.patch_stride,),
                    padding='VALID',
                    dtype=self.dtype,
                    name='stem_conv'
                )(x_emb)
                x_emb = nn.LayerNorm(
                    dtype=self.dtype,
                    name='stem_ln'
                )(x_emb)
            else: # Downsampling layers
                N, _ = x_emb.shape[1], x_emb.shape[2]
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    # 若要对齐官方实现，需constant->edge：在时间维末尾用边界复制进行补齐（而非常数0填充）
                    x_emb = jnp.pad(x_emb, ((0, 0), (0, pad_len), (0, 0)), mode='edge')

                x_emb = nn.LayerNorm(
                    dtype=self.dtype,
                    name=f'downsample_ln_{i}'
                )(x_emb)
                x_emb = nn.Conv(
                    features=self.dims[i],
                    kernel_size=(self.downsample_ratio,),
                    strides=(self.downsample_ratio,),
                    padding='VALID',
                    dtype=self.dtype,
                    name=f'downsample_conv_{i}'
                )(x_emb)

            # (B*M, N_i, D_i) -> (B, M, N_i, D_i) -> (B, M, D_i, N_i)
            _, N_i, D_i = x_emb.shape
            z = x_emb.reshape(B, M, N_i, D_i).transpose((0, 1, 3, 2))
            
            z = ModernTCNStage(
                num_blocks=self.num_blocks[i],
                d_model=self.dims[i],
                large_kernel_size=self.large_kernel_sizes[i],
                small_kernel_size=self.small_kernel_sizes[i],
                ffn_ratio=self.ffn_ratio,
                dropout_rate=self.dropout_rate,
                norm_type=self.norm_type,
                dtype=self.dtype,
                name=f'stage_{i}'
            )(z, training=training)
            # For next iteration, reshape and transpose back
            # (B, M, D_i, N_i) -> (B, M, N_i, D_i) -> (B*M, N_i, D_i)
            x_emb = z.transpose((0, 1, 3, 2)).reshape(B*M, N_i, D_i)
        # z: (B, M, D_final, N_final)
        _, _, D_final, N_final = z.shape
        
        # Post-processing: 对时间维度进行后处理以减少特征图的时间复杂度
        if self.post_proc == 'max_pool':
            # 在时间维度上进行最大池化 (窗口大小为2)
            # 使用 reduce_window 进行更精确的控制
            z = lax.reduce_window(
                z, 
                -jnp.inf, 
                lax.max,
                window_dimensions=(1, 1, 1, 2),
                window_strides=(1, 1, 1, 2),
                padding='VALID'
            )
        elif self.post_proc == 'avg_pool':
            # 在时间维度上进行平均池化 (窗口大小为2)
            pool_sum = lax.reduce_window(
                z,
                0.,
                lax.add,
                window_dimensions=(1, 1, 1, 2),
                window_strides=(1, 1, 1, 2),
                padding='VALID'
            )
            z = pool_sum / 2.0  # 除以窗口大小得到平均值
        elif self.post_proc == 'global_max_pool':
            # 全局最大池化：将时间维度完全压缩为1
            z = lax.reduce_window(
                z,
                -jnp.inf,
                lax.max,
                window_dimensions=(1, 1, 1, N_final),
                window_strides=(1, 1, 1, 1),
                padding='VALID'
            )
        elif self.post_proc == 'global_avg_pool':
            # 全局平均池化：将时间维度完全压缩为1
            pool_sum = lax.reduce_window(
                z,
                0.,
                lax.add,
                window_dimensions=(1, 1, 1, N_final),
                window_strides=(1, 1, 1, 1),
                padding='VALID'
            )
            z = pool_sum / N_final  # 除以窗口大小得到平均值
        # elif self.post_proc == 'none': 不做任何处理
        
        return z # Return the final feature map

class ModernTCN(nn.Module):
    """
    ModernTCN 预测模型 (Forecaster)。
    在内部使用 ModernTCNEncoder 提取特征，然后通过一个预测头输出预测结果。
    """
    # Encoder Hyperparameters
    n_vars: int
    patch_size: int
    patch_stride: int
    dims: Sequence[int]
    num_blocks: Sequence[int]
    large_kernel_sizes: Sequence[int]
    small_kernel_sizes: Sequence[int]
    downsample_ratio: int
    ffn_ratio: int = 2
    dropout_rate: float = 0.1
    revin: bool = False
    norm_type: Literal['batch', 'layer'] = 'layer'
    post_proc: Literal['none', 'max_pool', 'avg_pool', 'global_max_pool', 'global_avg_pool'] = 'none'
    
    # Head Hyperparameters
    target_window: int = 96
    individual: bool = False
    head_dropout: float = 0.1
    
    # Global dtype
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x: Array, training: bool = True) -> Array:
        encoder = ModernTCNEncoder(
            n_vars=self.n_vars,
            patch_size=self.patch_size,
            patch_stride=self.patch_stride,
            dims=self.dims,
            num_blocks=self.num_blocks,
            large_kernel_sizes=self.large_kernel_sizes,
            small_kernel_sizes=self.small_kernel_sizes,
            downsample_ratio=self.downsample_ratio,
            ffn_ratio=self.ffn_ratio,
            dropout_rate=self.dropout_rate,
            revin=self.revin,
            norm_type=self.norm_type,
            post_proc=self.post_proc,
            dtype=self.dtype,
            name='encoder'
        )
        features = encoder(x, training=training)

        # Calculate head_nf from encoder's output shape
        _, _, D_final, N_final = features.shape
        head_nf = D_final * N_final
        
        head = FlattenHead(
            n_vars=self.n_vars,
            head_nf=head_nf,
            target_window=self.target_window,
            individual=self.individual,
            head_dropout=self.head_dropout,
            dtype=self.dtype,
            name='head'
        )
        
        return head(features, training=training)
