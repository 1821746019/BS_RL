from functools import partial
from typing import Optional
import jax
import jax.numpy as np
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax import random
from jax.numpy.linalg import eigh


class SequenceLayer(nn.Module):
    """ Defines a single S5 layer, with S5 SSM, nonlinearity, etc.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                    we usually refer to this size as H
            activation  (string):   Type of activation function to use
            prenorm     (bool):     apply prenorm if true or postnorm if false
            step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                    e.g. after training on a different resolution for
                                    the speech commands benchmark
    """
    ssm: nn.Module
    d_model: int
    activation: str = "gelu"
    do_norm: bool = True
    prenorm: bool = True
    do_gtrxl_norm: bool = True
    step_rescale: float = 1.0

    def setup(self):
        """Initializes the ssm, layer norm and dense layers
        """
        self.seq = self.ssm(step_rescale=self.step_rescale)

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)

        self.norm = nn.LayerNorm()

    def __call__(self, hidden, x, d):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
             d (bool): reset signal (L,)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        if self.prenorm and self.do_norm:
            x = self.norm(x)
        # hidden, x = self.seq(hidden, x, d)
        hidden, x = jax.vmap(self.seq, in_axes=1, out_axes=1)(hidden, x, d)
        # hidden = jnp.swapaxes(hidden, 1, 0)
        if self.do_gtrxl_norm:
            x = self.norm(x)

        if self.activation in ["full_glu"]:
            x = nn.gelu(x)
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
        elif self.activation in ["half_glu1"]:
            x = nn.gelu(x)
            x = x * jax.nn.sigmoid(self.out2(x))
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = nn.gelu(x)
            x = x * jax.nn.sigmoid(self.out2(x1))
        elif self.activation in ["gelu"]:
            x = nn.gelu(x)
        else:
            raise NotImplementedError(
                   "Activation: {} not implemented".format(self.activation))

        x = skip + x
        if not self.prenorm and self.do_norm:
            x = self.norm(x)
        return hidden, x

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return jnp.zeros((1, batch_size, hidden_size), dtype=jnp.complex64)

def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(key, shape):
        """ Init function
             Args:
                 key: jax random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return random.uniform(key, shape) * (
            np.log(dt_max) - np.log(dt_min)
        ) + np.log(dt_min)

    return init


def init_log_steps(key, input):
    """ Initialize an array of learnable timescale parameters
         Args:
             key: jax random key
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of timescales (float32): (H,)
     """
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)

    return np.array(log_steps)


def init_VinvB(init_fun, rng, shape, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = VinvB.real
    VinvB_imag = VinvB.imag
    return np.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             key: jax random key
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = random.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return np.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             rng:       jax random key to be used with init function.
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = CV.real
    CV_imag = CV.imag
    return np.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using bilinear transform method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])

    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """ Discretize a diagonalized, continuous-time linear SSM
        using zero-order hold method.
        Args:
            Lambda (complex64): diagonal state matrix              (P,)
            B_tilde (complex64): input matrix                      (P, H)
            Delta (float32): discretization step sizes             (P,)
        Returns:
            discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = np.ones(Lambda.shape[0])
    Lambda_bar = np.exp(Lambda * Delta)
    B_bar = (1/Lambda * (Lambda_bar-Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


# Parallel scan operations
@jax.vmap
def binary_operator(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j

# Parallel scan operations
@jax.vmap
def binary_operator_reset(q_i, q_j):
    """ Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
        Args:
            q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
            q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
        Returns:
            new element ( A_out, Bu_out )
    """
    A_i, b_i, c_i = q_i
    A_j, b_j, c_j = q_j
    return (
        (A_j * A_i)*(1 - c_j) + A_j * c_j,
        (A_j * b_i + b_j)*(1 - c_j) + b_j * c_j,
        c_i * (1 - c_j) + c_j,
    )



def apply_ssm(Lambda_bar, B_bar, C_tilde, hidden, input_sequence, resets, conj_sym, bidirectional):
    """ Compute the LxH output of discretized SSM given an LxH input.
        Args:
            Lambda_bar (complex64): discretized diagonal state matrix    (P,)
            B_bar      (complex64): discretized input matrix             (P, H)
            C_tilde    (complex64): output matrix                        (H, P)
            input_sequence (float32): input sequence of features         (L, H)
            reset      (bool): input sequence of features                (L,)
            conj_sym (bool):         whether conjugate symmetry is enforced
            bidirectional (bool):    whether bidirectional setup is used,
                                  Note for this case C_tilde will have 2P cols
        Returns:
            ys (float32): the SSM outputs (S5 layer preactivations)      (L, H)
    """
    Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0],
                                            Lambda_bar.shape[0]))
    Bu_elements = jax.vmap(lambda u: B_bar @ u)(input_sequence)

    Lambda_elements = jnp.concatenate([
        jnp.ones((1, Lambda_bar.shape[0])),
        Lambda_elements,
    ])

    Bu_elements = jnp.concatenate([
        hidden,
        Bu_elements,
    ])

    if resets is None:
        _, xs = jax.lax.associative_scan(binary_operator, (Lambda_elements, Bu_elements))
    else:
        resets = jnp.concatenate([
            jnp.zeros(1),
            resets,
        ])
        _, xs, _ = jax.lax.associative_scan(binary_operator_reset, (Lambda_elements, Bu_elements, resets))
    xs = xs[1:]

    if conj_sym:
        return xs[np.newaxis, -1], jax.vmap(lambda x: 2*(C_tilde @ x).real)(xs)
    else:
        return xs[np.newaxis, -1], jax.vmap(lambda x: (C_tilde @ x).real)(xs)


class S5SSM(nn.Module):
    Lambda_re_init: jnp.ndarray
    Lambda_im_init: jnp.ndarray
    V: jnp.ndarray
    Vinv: jnp.ndarray

    H: int
    P: int
    C_init: str
    discretization: str
    dt_min: float
    dt_max: float
    conj_sym: bool = True
    clip_eigs: bool = False
    bidirectional: bool = False
    step_rescale: float = 1.0

    """ The S5 SSM
        Args:
            Lambda_re_init (complex64): Real part of init diag state matrix  (P,)
            Lambda_im_init (complex64): Imag part of init diag state matrix  (P,)
            V           (complex64): Eigenvectors used for init           (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init   (P,P)
            H           (int32):     Number of features of input seq 
            P           (int32):     state size
            C_init      (string):    Specifies How C is initialized
                         Options: [trunc_standard_normal: sample from truncated standard normal 
                                                        and then multiply by V, i.e. C_tilde=CV.
                                   lecun_normal: sample from Lecun_normal and then multiply by V.
                                   complex_normal: directly sample a complex valued output matrix 
                                                    from standard normal, does not multiply by V]
            conj_sym    (bool):    Whether conjugate symmetry is enforced
            clip_eigs   (bool):    Whether to enforce left-half plane condition, i.e.
                                   constrain real part of eigenvalues to be negative. 
                                   True recommended for autoregressive task/unbounded sequence lengths
                                   Discussed in https://arxiv.org/pdf/2206.11893.pdf.
            bidirectional (bool):  Whether model is bidirectional, if True, uses two C matrices
            discretization: (string) Specifies discretization method 
                             options: [zoh: zero-order hold method,
                                       bilinear: bilinear transform]
            dt_min:      (float32): minimum value to draw timescale values from when 
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when 
                                    initializing log_step
            step_rescale:  (float32): allows for uniformly changing the timescale parameter, e.g. after training 
                                    on a different resolution for the speech commands benchmark
    """

    def setup(self):
        """Initializes parameters once and performs discretization each time
           the SSM is applied to a sequence
        """

        if self.conj_sym:
            # Need to account for case where we actually sample real B and C, and then multiply
            # by the half sized Vinv and possibly V
            local_P = 2*self.P 
        else:
            local_P = self.P

        # Initialize diagonal state to state matrix Lambda (eigenvalues)
        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        # Initialize input to state (B) matrix
        B_init = lecun_normal()
        # 【修正】B 的形状应该基于存储的复数维度 P，而不是有效的实数维度 local_P。local_P 仅用于概念上的理解，不应参与 B 的形状定义
        B_shape = (self.P, self.H)
        self.B = self.param("B",
                            lambda rng, shape: init_VinvB(B_init,
                                                          rng,
                                                          shape,
                                                          self.Vinv),
                            B_shape)
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        # Initialize state to output (C) matrix
        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, self.P, 2) # 【修正】C 的形状也应该基于存储的复数维度 P
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, self.P, 2) # 【修正】C 的形状也应该基于存储的复数维度 P
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                   "C_init method {} not implemented".format(self.C_init))

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

            else:
                C = self.param("C", C_init, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]

        else:
            if self.bidirectional:
                self.C1 = self.param("C1",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)
                self.C2 = self.param("C2",
                                     lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                     C_shape)

                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = np.concatenate((C1, C2), axis=-1)

            else:
                self.C = self.param("C",
                                    lambda rng, shape: init_CV(C_init, rng, shape, self.V),
                                    C_shape)

                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        # Initialize feedthrough (D) matrix
        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        # Initialize learnable discretization timescale value
        self.log_step = self.param("log_step",
                                   init_log_steps,
                                   (self.P, self.dt_min, self.dt_max))
        step = self.step_rescale * np.exp(self.log_step[:, 0])

        # Discretize
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError("Discretization method {} not implemented".format(self.discretization))

    def __call__(self, hidden, input_sequence, resets):
        """
        Compute the LxH output of the S5 SSM given an LxH input sequence
        using a parallel scan.
        Args:
             input_sequence (float32): input sequence (L, H)
             resets (bool): input sequence (L,)
        Returns:
            output sequence (float32): (L, H)
        """
        hidden, ys = apply_ssm(self.Lambda_bar,
                       self.B_bar,
                       self.C_tilde,
                       hidden,
                       input_sequence,
                       resets,
                       self.conj_sym,
                       self.bidirectional)
        # Add feedthrough matrix output Du;
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return hidden, ys + Du


def init_S5SSM(H,
               P,
               Lambda_re_init,
               Lambda_im_init,
               V,
               Vinv,
               C_init,
               discretization,
               dt_min,
               dt_max,
               conj_sym,
               clip_eigs,
               bidirectional
               ):
    """Convenience function that will be used to initialize the SSM.
       Same arguments as defined in S5SSM above."""
    return partial(S5SSM,
                   H=H,
                   P=P,
                   Lambda_re_init=Lambda_re_init,
                   Lambda_im_init=Lambda_im_init,
                   V=V,
                   Vinv=Vinv,
                   C_init=C_init,
                   discretization=discretization,
                   dt_min=dt_min,
                   dt_max=dt_max,
                   conj_sym=conj_sym,
                   clip_eigs=clip_eigs,
                   bidirectional=bidirectional)


def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig

class StackedEncoderModel(nn.Module):
    """ Defines a stack of S5 layers to be used as an encoder.
        Args:
            ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
            d_model     (int32):    this is the feature size of the layer inputs and outputs
                                     we usually refer to this size as H
            n_layers    (int32):    the number of S5 layers to stack
            activation  (string):   Type of activation function to use
            prenorm     (bool):     apply prenorm if true or postnorm if false
    """
    ssm: nn.Module
    d_model: int
    n_layers: int
    activation: str = "gelu"
    do_norm: bool = True
    prenorm: bool = True
    do_gtrxl_norm: bool = True

    def setup(self):
        """
        Initializes a linear encoder and the stack of S5 layers.
        """
        # self.encoder = nn.Dense(self.d_model)
        self.layers = [
            SequenceLayer(
                ssm=self.ssm,
                d_model=self.d_model,
                activation=self.activation,
                do_norm=self.do_norm,
                prenorm=self.prenorm,
                do_gtrxl_norm=self.do_gtrxl_norm,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, hidden, x, d):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output sequence (float32): (L, d_model)
        """
        # x = self.encoder(x)
        new_hiddens = []
        for i, layer in enumerate(self.layers):
            new_h, x = layer(hidden[i], x, d)
            new_hiddens.append(new_h)
    
        return new_hiddens, x

    @staticmethod
    def initialize_carry(batch_size, hidden_size, n_layers):
        # Use a dummy key since the default state init fn is just zeros.
        return [jnp.zeros((1, batch_size, hidden_size), dtype=jnp.complex64) for _ in range(n_layers)]


class S5Summarizer(nn.Module):
    hidden_dim: int
    num_layers: int = 1
    delta_min: float = 0.001
    delta_max: float = 0.1

    def initialize_state(self, batch_size: int):
        """Initializes the real-valued carry state."""
        return jnp.zeros((self.num_layers, batch_size, self.hidden_dim), dtype=jnp.float32)
        
    def setup(self):
        H = self.hidden_dim
        if H % 2 != 0:
            raise ValueError(f"S5Summarizer.hidden_dim must be even for conjugate symmetry, got {H}")
        P = H // 2  # ensure H = 2P to pack complex state
        # Build DPLR HiPPO
        Lambda, _, _, V, _ = make_DPLR_HiPPO(P)
        Vinv = jnp.conj(jnp.transpose(V))
        ssm_init_fn = init_S5SSM(
            H=H,
            P=P,
            Lambda_re_init=jnp.real(Lambda),
            Lambda_im_init=jnp.imag(Lambda),
            V=V,
            Vinv=Vinv,
            C_init="lecun_normal",
            discretization="zoh",
            dt_min=self.delta_min,
            dt_max=self.delta_max,
            conj_sym=True, 
            clip_eigs=False,
            bidirectional=False,
        )
        """
conj_sym是什么？
S5模型内部的状态演化由一个复杂的线性系统描述。为了更好地分析和计算这个系统，模型会将其分解到“特征模式”（即特征向量和特征值）。对于一个用真实世界数值定义的系统，它的复数特征值必然以共轭对的形式存在 。
为什么能优化？
减少存储和计算：既然这些特征值是成对的，我们只需要存储和计算其中一半（例如，虚部为正的那一半） 。
效率翻倍：通过只处理一半的数据，模型的运行时长和内存占用几乎可以减半 。这是一个巨大的性能提升，尤其是在处理长序列和大型模型时。
保证实数输出：这个操作的另一个重要作用是，它能确保模型的最终输出是实数，而不是复数，这对于神经网络的后续层是必需的 。   
总结：conj_sym(共轭对称)是S5模型中的一个关键效率开关。当它开启时：
利用了“共轭对称”这一数学性质 。
只存储和计算一半的复数状态 。
将模型运行速度和内存效率提升近一倍 。
确保模型输出为实数，保证了计算的正确性 。
        """
        self.encoder = nn.Dense(H, name="in_proj")
        self.s5_stack = StackedEncoderModel(
            ssm=ssm_init_fn,
            d_model=H,
            n_layers=self.num_layers,
            activation="full_glu",
            do_norm=True,
            prenorm=True,
            do_gtrxl_norm=True,
        )

    def _pack_hidden_real_to_complex(self, h_real: jnp.ndarray) -> jnp.ndarray:
        """h_real: [B, H] -> complex [1, B, P] where H=2P"""
        H = h_real.shape[-1]
        P = H // 2
        real = h_real[..., :P]
        imag = h_real[..., P: P * 2]
        h_complex = real + 1j * imag
        return h_complex[None, ...]

    def _unpack_hidden_complex_to_real(self, h_complex: jnp.ndarray) -> jnp.ndarray:
        """h_complex: [1, B, P] -> real [B, H] with H=2P"""
        real = jnp.real(h_complex[0])
        imag = jnp.imag(h_complex[0])
        return jnp.concatenate([real, imag], axis=-1)

    @nn.compact
    def __call__(self, x: jnp.ndarray, initial_state: Optional[jnp.ndarray] = None):
        """
        x: [B, T, D]
        Returns:
          outputs: [B, T, H]
          final_state: (h) with shapes [L,B,H]
        """
        B, T, _ = x.shape
        H = self.hidden_dim
        L = self.num_layers

        x_proj = self.encoder(x)  # [B, T, H]
        # Transpose to [T, B, H] for S5 stack
        x_tbH = jnp.swapaxes(x_proj, 0, 1)

        # Prepare initial hidden states per S5 layer as complex [1,B,P]
        if initial_state is None:
            initial_state = self.initialize_state(B)
        
        h_init = initial_state  # [L, B, H]
        h0_layers = [self._pack_hidden_real_to_complex(h_init[i]) for i in range(L)]

        # Resets: zeros [T, B]
        d_tb = jnp.zeros((T, B), dtype=jnp.float32)

        # Apply S5 stack
        new_h_layers, y_tbH = self.s5_stack(h0_layers, x_tbH, d_tb)

        # Convert new hidden back to real [L, B, H]
        final_h_real = jnp.stack([self._unpack_hidden_complex_to_real(hc) for hc in new_h_layers], axis=0)
        
        y_bTH = jnp.swapaxes(y_tbH, 0, 1)
        outputs = y_bTH
        return outputs, final_h_real

import warnings, jax.numpy as jnp
warnings.filterwarnings("ignore", category=jnp.ComplexWarning)
"""
/root/project/.venv/lib/python3.11/site-packages/jax/_src/lax/lax.py:5125: ComplexWarning: Casting complex values to real discards the imaginary part
  x_bar = _convert_element_type(x_bar, x.aval.dtype, x.aval.weak_type)
不会影响训练的正确性。
这是 JAX 在反向传播里把复数型共轭梯度投回到实数参数时的提示，不是报错，训练会继续进行。
原因：S5 内部用复数态计算，但你网络的参数是以“实数对”（real, imag）存储并参与优化，最终输出传给后续网络都是实数。JAX 的梯度在回写到实数参数时会丢弃虚部，因此给出 ComplexWarning，但这在数学上等价于对实部/虚部分别求导，不影响优化正确性。
关键依据（代码）
S5 的输出在前向里已显式取实部，确保传给下游网络是实数：
S5 的隐藏状态用实数组拼成复数（forward）/再拆回实数（state 接口），参数本身是实数张量：
何时需要担心
只有当你真的把 Flax 参数直接定义为复数 dtype，并希望其“虚部”也被单独优化时，丢弃虚部才会带来信息损失。现在你的实现用的是实数参数对（最后一维 size=2），不受影响。
若在 loss 中直接保留了复数（未取 real），那会导致更严重的问题（NaN/报错），但你当前实现已在关键处取实部。
"""