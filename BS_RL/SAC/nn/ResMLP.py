import flax.linen as nn
import jax.numpy as jnp
from typing import List, Callable, Union, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass, field
import json

class ResidualStrategy(Enum):
    """残差连接策略枚举"""
    PROJECTION = "projection"      # 线性投影
    CONV = "conv"                 # 1x1卷积投影
    DROP = "drop"                 # 维度不匹配时放弃残差连接
    IDENTITY = "identity"         # 仅维度匹配时使用残差连接
    NONE = "none"                 # 完全不使用残差连接

class ActivationPosition(Enum):
    """激活函数位置枚举"""
    PRE = "pre"                   # 预激活（LayerNorm -> Activation -> Dense）
    POST = "post"                 # 后激活（Dense -> Activation）
    BOTH = "both"                 # 前后都有激活

@dataclass
class ResMLPConfig:
    """ResMLP配置类
    
    提供了完整的配置选项和预设配置，支持序列化和验证。
    """
    # 基本结构配置
    hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 128])
    
    # 残差连接配置
    residual_strategy: Union[str, ResidualStrategy] = ResidualStrategy.PROJECTION
    
    # 激活和归一化配置
    activation_position: Union[str, ActivationPosition] = ActivationPosition.PRE
    use_layer_norm: bool = True
    add_initial_embedding_layer: bool = True
    skip_final_ln: bool = False
    
    # 投影配置
    projection_bias: bool = True
    dense_bias: bool = True
    
    # 正则化配置
    dropout_rate: float = 0.0
    
    # 高级特性配置
    use_glu: bool = False          # Gated Linear Units
    use_highway: bool = False      # Highway connections
    
    # 初始化配置
    kernel_init: str = "lecun_normal"  # xavier_normal, he_normal, lecun_normal
    bias_init: str = "zeros"           # zeros, normal
    
    # 元信息
    name: Optional[str] = None
    description: Optional[str] = None
    
    def __post_init__(self):
        """后处理：类型转换和验证"""
        # 转换枚举类型
        if isinstance(self.residual_strategy, str):
            self.residual_strategy = ResidualStrategy(self.residual_strategy)
        if isinstance(self.activation_position, str):
            self.activation_position = ActivationPosition(self.activation_position)
            
        # 验证配置
        self.validate()
    
    def validate(self):
        """验证配置的合理性"""
        if not self.hidden_dims:
            raise ValueError("hidden_dims不能为空")
        
        if any(dim <= 0 for dim in self.hidden_dims):
            raise ValueError("所有hidden_dims必须为正数")
            
        if not 0.0 <= self.dropout_rate <= 1.0:
            raise ValueError("dropout_rate必须在[0, 1]范围内")
            
        # 特定组合的警告
        if (self.residual_strategy == ResidualStrategy.CONV and 
            self.use_highway):
            print("警告: CONV投影与Highway连接同时使用可能导致参数冗余")
            
        # skip_final_ln只在有post-activation时有用
        if (self.skip_final_ln and 
            self.activation_position == ActivationPosition.PRE):
            print("警告: skip_final_ln在PRE激活模式下无效，因为没有post LayerNorm")
            
        # add_initial_embedding_layer只在有pre-activation时有用  
        if (self.add_initial_embedding_layer and 
            self.activation_position == ActivationPosition.POST):
            print("警告: add_initial_embedding_layer在POST激活模式下无效，因为没有pre LayerNorm")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典，用于序列化"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ResMLPConfig':
        """从字典创建配置"""
        return cls(**config_dict)
    
    def save_json(self, filepath: str):
        """保存配置到JSON文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_json(cls, filepath: str) -> 'ResMLPConfig':
        """从JSON文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)
    
    def copy(self, **overrides) -> 'ResMLPConfig':
        """创建配置副本并覆盖指定参数"""
        config_dict = self.to_dict()
        config_dict.update(overrides)
        return self.__class__.from_dict(config_dict)

# 预设配置
class ResMLPPresets:
    """预设配置集合"""
    
    @staticmethod
    def standard() -> ResMLPConfig:
        """标准配置：平衡性能和效率"""
        return ResMLPConfig(
            hidden_dims=[128, 256, 512, 256, 128],
            residual_strategy=ResidualStrategy.PROJECTION,
            activation_position=ActivationPosition.PRE,
            use_layer_norm=True,
            dropout_rate=0.1,
            name="standard",
            description="标准ResMLP配置，平衡性能和效率"
        )
    
    @staticmethod
    def lightweight() -> ResMLPConfig:
        """轻量级配置：更少参数，更快推理"""
        return ResMLPConfig(
            hidden_dims=[64, 128, 64],
            residual_strategy=ResidualStrategy.DROP,
            activation_position=ActivationPosition.POST,
            use_layer_norm=False,
            dropout_rate=0.0,
            projection_bias=False,
            name="lightweight",
            description="轻量级配置，适合资源受限环境"
        )
    
    @staticmethod
    def high_capacity() -> ResMLPConfig:
        """高容量配置：最大表达能力"""
        return ResMLPConfig(
            hidden_dims=[256, 512, 1024, 512, 256],
            residual_strategy=ResidualStrategy.PROJECTION,
            activation_position=ActivationPosition.PRE,
            use_layer_norm=True,
            dropout_rate=0.15,
            use_glu=True,
            use_highway=True,
            name="high_capacity",
            description="高容量配置，最大化模型表达能力"
        )
    
    @staticmethod
    def conv_based() -> ResMLPConfig:
        """基于卷积的配置：利用空间归纳偏置"""
        return ResMLPConfig(
            hidden_dims=[128, 256, 512, 256],
            residual_strategy=ResidualStrategy.CONV,
            activation_position=ActivationPosition.PRE,
            use_layer_norm=True,
            dropout_rate=0.1,
            name="conv_based",
            description="使用卷积投影的配置，适合有空间结构的数据"
        )
    
    @staticmethod
    def vanilla_mlp() -> ResMLPConfig:
        """普通MLP配置：无残差连接的基线"""
        return ResMLPConfig(
            hidden_dims=[128, 256, 512, 256, 128],
            residual_strategy=ResidualStrategy.NONE,
            activation_position=ActivationPosition.POST,
            use_layer_norm=False,
            dropout_rate=0.0,
            name="vanilla_mlp",
            description="普通MLP，无残差连接，用作基线对比"
        )

class UnifiedResMLP(nn.Module):
    """统一的可配置ResMLP实现
    
    使用ResMLPConfig进行配置，支持多种残差连接策略和激活模式。
    """
    config: ResMLPConfig
    activation: Callable = nn.relu
    
    def setup(self):
        """模块初始化"""
        # 选择初始化函数
        if self.config.kernel_init == "xavier_normal":
            self.kernel_init_fn = nn.initializers.xavier_normal()
        elif self.config.kernel_init == "he_normal":
            self.kernel_init_fn = nn.initializers.he_normal()
        else:  # lecun_normal
            self.kernel_init_fn = nn.initializers.lecun_normal()
            
        if self.config.bias_init == "normal":
            self.bias_init_fn = nn.initializers.normal(stddev=0.01)
        else:  # zeros
            self.bias_init_fn = nn.initializers.zeros
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True):
        """前向传播
        
        Args:
            x: 输入张量 [batch_size, ..., input_dim]
            training: 是否处于训练模式
            
        Returns:
            输出张量 [batch_size, ..., hidden_dims[-1]]
        """
        
        for i, dim in enumerate(self.config.hidden_dims):
            if i == 0 and self.config.add_initial_embedding_layer:
                input_x = nn.Dense(dim, kernel_init=self.kernel_init_fn, bias_init=self.bias_init_fn, name=f"embedding_{i}")(x)
            else:
                input_x = x
            
            # 计算残差连接的shortcut
            shortcut = self._compute_shortcut(input_x, dim, i)
            
            # 主分支计算
            y = self._forward_branch(input_x, dim, i, training)
            
            # 应用残差连接
            x = self._apply_residual(shortcut, y, input_x, dim)
            
        return x
    
    def _compute_shortcut(self, x: jnp.ndarray, target_dim: int, layer_idx: int):
        """计算残差连接的shortcut"""
        if self.config.residual_strategy == ResidualStrategy.NONE:
            return None
            
        input_dim = x.shape[-1]
        
        if input_dim == target_dim:
            # 维度匹配，直接使用identity shortcut
            return x
            
        elif self.config.residual_strategy == ResidualStrategy.PROJECTION:
            # 使用线性投影
            return nn.Dense(
                target_dim, 
                use_bias=self.config.projection_bias,
                kernel_init=self.kernel_init_fn,
                bias_init=self.bias_init_fn,
                name=f"shortcut_projection_{layer_idx}"
            )(x)
            
        elif self.config.residual_strategy == ResidualStrategy.CONV:
            # 使用1x1卷积
            if len(x.shape) == 2:  # [batch, features]
                x_reshaped = x[..., None, None, :]  # [batch, 1, 1, features]
                conv_out = nn.Conv(
                    target_dim, 
                    kernel_size=(1, 1),
                    use_bias=self.config.projection_bias,
                    kernel_init=self.kernel_init_fn,
                    bias_init=self.bias_init_fn,
                    name=f"shortcut_conv_{layer_idx}"
                )(x_reshaped)
                return conv_out.squeeze(axis=(1, 2))
            else:
                return nn.Conv(
                    target_dim,
                    kernel_size=(1, 1),
                    use_bias=self.config.projection_bias,
                    kernel_init=self.kernel_init_fn,
                    bias_init=self.bias_init_fn,
                    name=f"shortcut_conv_{layer_idx}"
                )(x)
                
        elif self.config.residual_strategy in [ResidualStrategy.DROP, ResidualStrategy.IDENTITY]:
            return None
            
        else:
            raise ValueError(f"未知的残差策略: {self.config.residual_strategy}")
    
    def _forward_branch(self, x: jnp.ndarray, target_dim: int, layer_idx: int, training: bool):
        """计算主分支"""
        y = x
        
        # Pre-activation
        if self.config.activation_position in [ActivationPosition.PRE, ActivationPosition.BOTH]:
            if (self.config.use_layer_norm):
                y = nn.LayerNorm(name=f"pre_norm_{layer_idx}")(y)
            y = self.activation(y)
        
        # Dropout
        if self.config.dropout_rate > 0.0:
            y = nn.Dropout(self.config.dropout_rate, deterministic=not training)(y)
        
        # 主要的Dense层
        if self.config.use_glu:
            # Gated Linear Unit
            gate = nn.Dense(
                target_dim, 
                use_bias=self.config.dense_bias,
                kernel_init=self.kernel_init_fn,
                bias_init=self.bias_init_fn,
                name=f"gate_{layer_idx}"
            )(y)
            value = nn.Dense(
                target_dim, 
                use_bias=self.config.dense_bias,
                kernel_init=self.kernel_init_fn,
                bias_init=self.bias_init_fn,
                name=f"value_{layer_idx}"
            )(y)
            y = nn.sigmoid(gate) * value
        else:
            # 标准Dense层
            y = nn.Dense(
                target_dim, 
                use_bias=self.config.dense_bias,
                kernel_init=self.kernel_init_fn,
                bias_init=self.bias_init_fn,
                name=f"dense_{layer_idx}"
            )(y)
        
        # Highway connection (可选)
        if self.config.use_highway and x.shape[-1] == target_dim:
            gate = nn.Dense(
                target_dim,
                kernel_init=self.kernel_init_fn,
                bias_init=nn.initializers.constant(-1),  # Highway gate bias
                name=f"highway_gate_{layer_idx}"
            )(x)
            gate = nn.sigmoid(gate)
            y = gate * y + (1 - gate) * x
        
        # Post-activation
        if self.config.activation_position in [ActivationPosition.POST, ActivationPosition.BOTH]:
            y = self.activation(y)
            if (self.config.use_layer_norm and 
                not (self.config.skip_final_ln and layer_idx == len(self.config.hidden_dims) - 1)):
                y = nn.LayerNorm(name=f"post_norm_{layer_idx}")(y)
        
        return y
    
    def _apply_residual(self, shortcut, y, input_x: jnp.ndarray, target_dim: int):
        """应用残差连接"""
        if self.config.residual_strategy == ResidualStrategy.NONE:
            return y
            
        if shortcut is not None:
            return shortcut + y
        else:
            if self.config.residual_strategy == ResidualStrategy.DROP:
                return y
            elif self.config.residual_strategy == ResidualStrategy.IDENTITY:
                if input_x.shape[-1] == target_dim:
                    return input_x + y
                else:
                    return y
            else:
                return y

# 便利的工厂函数
def create_resmlp(config: Union[ResMLPConfig, str], activation: Callable = nn.relu) -> UnifiedResMLP:
    """创建ResMLP模型
    
    Args:
        config: 配置对象或预设名称
        activation: 激活函数
        
    Returns:
        配置好的UnifiedResMLP模型
    """
    if isinstance(config, str):
        # 从预设创建
        preset_map = {
            "standard": ResMLPPresets.standard,
            "lightweight": ResMLPPresets.lightweight,
            "high_capacity": ResMLPPresets.high_capacity,
            "conv_based": ResMLPPresets.conv_based,
            "vanilla_mlp": ResMLPPresets.vanilla_mlp,
        }
        if config not in preset_map:
            raise ValueError(f"未知预设: {config}. 可用预设: {list(preset_map.keys())}")
        config = preset_map[config]()
    
    return UnifiedResMLP(config=config, activation=activation)

# 使用示例
if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    
    # 1. 使用预设配置
    model_standard = create_resmlp("standard")
    
    # 2. 使用自定义配置
    custom_config = ResMLPConfig(
        hidden_dims=[64, 128, 256, 128],
        residual_strategy=ResidualStrategy.PROJECTION,
        dropout_rate=0.1,
        use_glu=True,
        name="custom_model"
    )
    model_custom = create_resmlp(custom_config)
    
    # 3. 基于预设修改配置
    modified_config = ResMLPPresets.standard().copy(
        hidden_dims=[256, 512, 256],
        dropout_rate=0.2
    )
    model_modified = create_resmlp(modified_config)
    
    # 4. 配置序列化示例
    custom_config.save_json("resmlp_config.json")
    loaded_config = ResMLPConfig.load_json("resmlp_config.json")
    
    # 测试所有模型
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (32, 64))
    
    models = {
        "standard": model_standard,
        "custom": model_custom,
        "modified": model_modified
    }
    
    for name, model in models.items():
        key, subkey = jax.random.split(key)
        params = model.init(subkey, x)
        output = model.apply(params, x)
        param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
        
        print(f"{name}:")
        print(f"  配置: {model.config.name or 'unnamed'}")
        print(f"  输入: {x.shape} -> 输出: {output.shape}")
        print(f"  参数数量: {param_count:,}")
        print(f"  残差策略: {model.config.residual_strategy.value}")
        print()

# 配置管理最佳实践
"""
### 使用建议:

1. **项目初期**: 使用预设配置快速验证
   model = create_resmlp("standard")

2. **超参数调优**: 基于预设进行微调
   config = ResMLPPresets.standard().copy(dropout_rate=0.15)

3. **配置管理**: 使用JSON文件管理不同实验配置
   config.save_json("experiments/exp_001.json")

4. **生产部署**: 使用validated配置确保一致性
   config = ResMLPConfig.load_json("production_config.json")

### 扩展指南:

添加新配置项只需:
1. 在ResMLPConfig中添加字段
2. 在UnifiedResMLP中添加对应逻辑
3. 更新预设配置（可选）

这种设计保证了向后兼容性和可扩展性。
"""