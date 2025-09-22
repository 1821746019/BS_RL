
# --- 使用示例 ---
from BS_RL.nn.ModernTCN import ModernTCNEncoder
import jax
import jax.numpy as jnp
from jax import lax
from flax import linen as nn
from typing import Sequence, Optional, Literal
from BS_RL.nn.ModernTCN import ModernTCN


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    
    # 模拟 K 线数据 (Batch, SeqLen, Vars)
    batch_size = 16
    seq_len = 512
    n_vars = 7
    
    mock_kline_data = jnp.ones((batch_size, seq_len, n_vars))
    
    # --- 1. 使用 Encoder 提取特征 ---
    print("--- Testing ModernTCNEncoder ---")
    encoder_params = {
        'n_vars': n_vars,
        'patch_size': 16,
        'patch_stride': 8,
        'dims': [64, 128, 256],
        'num_blocks': [3, 3, 3],
        'large_kernel_sizes': [31, 31, 31],
        'small_kernel_sizes': [5, 5, 5],
        'downsample_ratio': 2,
        'ffn_ratio': 2,
        'dropout_rate': 0.1,
        'norm_type': 'layer',
        'post_proc': 'max_pool',
    }
    encoder = ModernTCNEncoder(**encoder_params)
    encoder_vars = encoder.init({'params': key, 'dropout': key}, mock_kline_data, training=False)
    features = encoder.apply(encoder_vars, mock_kline_data, training=False)
    
    print("输入 K-Line 数据形状:", mock_kline_data.shape)
    print("编码后特征形状 (B, M, D, N):", features.shape)
    # Expected N before post_proc = (512 / 8) / 2 / 2 = 16
    # Expected N after max_pool = 16 / 2 = 8 (因为post_proc='max_pool', 窗口大小为2)
    # Expected D = 256 (last dim in `dims`)
    expected_feature_shape = (batch_size, n_vars, encoder_params['dims'][-1], 8)
    assert features.shape == expected_feature_shape
    print("Encoder 输出形状符合预期!\n")

    # --- 2. 使用 Forecaster (完整的 ModernTCN) 进行预测 ---
    print("--- Testing ModernTCN (Forecaster) ---")
    forecaster_params = {
        'n_vars': n_vars,
        'patch_size': 16,
        'patch_stride': 8,
        'dims': [64, 128, 256],
        'num_blocks': [3, 3, 3],
        'large_kernel_sizes': [31, 31, 31],
        'small_kernel_sizes': [5, 5, 5],
        'downsample_ratio': 2,
        'ffn_ratio': 2,
        'target_window': 96,
        'individual': False,
        'head_dropout': 0.1,
        'dropout_rate': 0.1,
        'norm_type': 'layer',
        'post_proc': 'max_pool',
    }
    forecaster = ModernTCN(**forecaster_params)
    forecaster_vars = forecaster.init({'params': key, 'dropout': key}, mock_kline_data, training=False)
    
    # 推理模式
    predictions = forecaster.apply(
        forecaster_vars,
        mock_kline_data,
        training=False
    )

    print("输入 K-Line 数据形状:", mock_kline_data.shape)
    print("输出预测形状 (B, target_window, M):", predictions.shape)

    # 验证输出形状是否符合预期
    expected_prediction_shape = (batch_size, forecaster_params['target_window'], n_vars)
    assert predictions.shape == expected_prediction_shape
    print("Forecaster 输出形状符合预期!")
    
    # --- 3. 测试不同的 post_proc 选项 ---
    print("\n--- Testing Different Post-processing Options ---")
    post_proc_options = ['none', 'max_pool', 'avg_pool', 'global_max_pool', 'global_avg_pool']
    
    for post_proc in post_proc_options:
        print(f"Testing post_proc='{post_proc}'...")
        test_encoder = ModernTCNEncoder(
            n_vars=n_vars,
            patch_size=16,
            patch_stride=8,
            dims=[64, 128, 256],
            num_blocks=[3, 3, 3],
            large_kernel_sizes=[31, 31, 31],
            small_kernel_sizes=[5, 5, 5],
            downsample_ratio=2,
            ffn_ratio=2,
            dropout_rate=0.1,
            norm_type='layer',
            post_proc=post_proc,
        )
        test_vars = test_encoder.init({'params': key, 'dropout': key}, mock_kline_data, training=False)
        test_features = test_encoder.apply(test_vars, mock_kline_data, training=False)
        
        expected_n_final = {
            'none': 16,
            'max_pool': 8,  # 16 // 2
            'avg_pool': 8,  # 16 // 2
            'global_max_pool': 1,
            'global_avg_pool': 1
        }[post_proc]
        
        expected_shape = (batch_size, n_vars, 256, expected_n_final)
        assert test_features.shape == expected_shape, f"post_proc='{post_proc}' failed: expected {expected_shape}, got {test_features.shape}"
        print(f"  ✓ post_proc='{post_proc}' 输出形状: {test_features.shape}")
    
    print("所有 post_proc 选项测试通过!")
"""
若我想分别编码1m级别的1h k线和1h级别的24*3h k线，应采用什么样的ModernTCNEncoder配置？为什么？
原理小结
最终特征的时间尺度由两件事决定：N_final 与每个 token 的有效步长 s_eff。
公式：N0 ≈ ceil(L / patch_stride)；N_final ≈ ceil(N0 / downsample_ratio^(num_stages-1))；s_eff = patch_stride × downsample_ratio^(num_stages-1)。
下面给出两套实用配置与理由（保留 LayerNorm，不开多尺度融合）。
1) 短期：1m 级别的 1h K 线（L=60）
目标：保留较高分辨率，聚焦近 30–60min 模式。
patch_size=4, patch_stride=2
dims=[64, 128], num_blocks=[2, 2]
large_kernel_sizes=[7, 9], small_kernel_sizes=[3, 3]
downsample_ratio=2, ffn_ratio=2, dropout_rate=0.1, norm_type='layer'
结果与原因：
N0 ≈ ceil(60/2)=30；num_stages=2 → N_final ≈ ceil(30/2)=15
s_eff = 2×2=4min；最后一层大核 9 token ≈ 9×4=36min 的感受野（配合前级累计，覆盖 ~半小时级别），细节不至于过度丢失
可更保守（更短期）：large_kernel_sizes=[5,7]；更激进（更平滑）：[9,11]
2) 中长期：1h 级别的 72h K 线（L=72）
目标：建模 1–7 天趋势，允许时间分辨率降低。
patch_size=8, patch_stride=2
dims=[64, 128, 256], num_blocks=[3, 3, 3]
large_kernel_sizes=[15, 21, 31], small_kernel_sizes=[3, 3, 5]
downsample_ratio=2, ffn_ratio=2, dropout_rate=0.1, norm_type='layer'
结果与原因：
N0 ≈ ceil(72/2)=36；num_stages=3 → N_final ≈ ceil(36/4)=9
s_eff = 2×4=8h；最后一层 31 token ≈ 31×8=248h 感受野（覆盖多日），中层/浅层核较小，兼顾中期结构
若你其实想编码“24 个 3h K 线”（L=24, 步长=3h），推荐：
patch_size=4, patch_stride=1；dims=[64,128,256]；num_blocks=[2,2,2]
large_kernel_sizes=[7,11,15]；small_kernel_sizes=[3,3,3]；downsample_ratio=2
N0≈24；N_final≈ceil(24/4)=6；s_eff=1×4=4 个 3h bar=12h；最后层约 15×12h=180h
选型提示
想更“短期”：减小 patch_stride、减少 stage 数或减小 large_kernel。
想更“长期”：增大 large_kernel、增加 stage 数；或略增 patch_stride。
两路并行使用：短期编码器喂微结构，长期编码器喂趋势，再在策略网络中融合。
"""

"""
那Encoder输出的数据还要做什么处理吗？还是直接展平后接MLP？
结论：可以直接展平接 MLP，但更稳妥的是“先做轻量聚合，再映射”。给你三个从简到强的推荐用法，按需选择。
基线1（最快速可用）
操作: 将 (B, M, D, N) 先转为 (B, M, D*N)，再展平为 (B, M*D*N)，接 MLP。
适用: 模型小、训练样本不多时先跑起来。
提示: 容易过拟合；可加 LayerNorm + Dropout 在展平后。
基线2（更稳健，推荐）
操作: 沿时间维做池化 (mean/max/attention)：(B, M, D, N) -> (B, M, D)；再展平 (B, M*D) 接 MLP。
含义: 显式“摘要”时间轴，减少参数和噪声，更利于 RL 稳定。
变体: 也可先用 1×1 conv/线性把 D 压到较小维度再展平。
基线3（按变量编码再融合，最通用）
操作: 类似你 Forecaster 的 FlattenHead 思路，对每个变量单独线性映射：(B, M, D, N) -> (B, M, D*N) -> Linear(E) 得到 (B, M, E)；再做变量级聚合（mean/sum/attention/gating），得 (B, E)；接策略/价值头。
优点: 保留“每个资产/因子”的表征，融合更可控；易于加入注意力做“变量重要性”。
实践建议
归一化: 展平或聚合后加 LayerNorm/RMSNorm 能显著稳住训练（你 batch 小，LN 合理）。
正则: 适量 Dropout(p=0.1) 或 WeightDecay 防过拟合。
无信息泄露: 你当前编码器仅用过去信息，末端“边界复制”不引入未来信息，在线可用。
多时间尺度并联: 若你用 1m 与 3h 两个 Encoder，分别得到 (B, E1)、(B, E2)，做 LayerNorm 后拼接 (B, E1+E2)，再接 MLP；或做门控融合。
向量尺度: 为稳定 RL，控制最终 state 向量在 256～1024 维区间较稳（太大容易难学、太小表征欠丰富）。
简短选择指南
要简单先跑: 选“基线1”。
要稳健抗噪: 选“基线2”（时间池化后展平）。
要保留变量粒度并可扩展到注意力: 选“基线3”（强烈推荐作为长期方案）。
"""