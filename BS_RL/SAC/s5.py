from functools import partial
import jax
import jax.numpy as jnp
from flax import linen as nn
from jax.nn.initializers import lecun_normal, normal
from jax.numpy.linalg import eigh


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    def init(key, shape):
        return jax.random.uniform(key, shape) * (
            jnp.log(dt_max) - jnp.log(dt_min)
        ) + jnp.log(dt_min)
    return init


def init_log_steps(key, input):
    H, dt_min, dt_max = input
    log_steps = []
    for i in range(H):
        key, skey = jax.random.split(key)
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(skey, shape=(1,))
        log_steps.append(log_step)
    return jnp.array(log_steps)


def init_VinvB(init_fun, rng, shape, Vinv):
    B = init_fun(rng, shape)
    VinvB = Vinv @ B
    VinvB_real = jnp.real(VinvB)
    VinvB_imag = jnp.imag(VinvB)
    return jnp.concatenate((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)


def trunc_standard_normal(key, shape):
    H, P, _ = shape
    Cs = []
    for i in range(H):
        key, skey = jax.random.split(key)
        C = lecun_normal()(skey, shape=(1, P, 2))
        Cs.append(C)
    return jnp.array(Cs)[:, 0]


def init_CV(init_fun, rng, shape, V):
    C_ = init_fun(rng, shape)
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    CV_real = jnp.real(CV)
    CV_imag = jnp.imag(CV)
    return jnp.concatenate((CV_real[..., None], CV_imag[..., None]), axis=-1)


def discretize_bilinear(Lambda, B_tilde, Delta):
    Identity = jnp.ones(Lambda.shape[0])
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    Identity = jnp.ones(Lambda.shape[0])
    Lambda_bar = jnp.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


@jax.vmap
def _binary_operator(q_i, q_j):
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


@jax.vmap
def _binary_operator_reset(q_i, q_j):
    A_i, b_i, c_i = q_i
    A_j, b_j, c_j = q_j
    return (
        (A_j * A_i) * (1 - c_j) + A_j * c_j,
        (A_j * b_i + b_j) * (1 - c_j) + b_j * c_j,
        c_i * (1 - c_j) + c_j,
    )


def apply_ssm(Lambda_bar, B_bar, C_tilde, hidden, input_sequence, resets, conj_sym, bidirectional):
    Lambda_elements = Lambda_bar * jnp.ones((input_sequence.shape[0], Lambda_bar.shape[0]))
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
        _, xs = jax.lax.associative_scan(_binary_operator, (Lambda_elements, Bu_elements))
    else:
        resets = jnp.concatenate([
            jnp.zeros(1),
            resets,
        ])
        _, xs, _ = jax.lax.associative_scan(_binary_operator_reset, (Lambda_elements, Bu_elements, resets))
    xs = xs[1:]

    if conj_sym:
        return xs[jnp.newaxis, -1], jax.vmap(lambda x: 2 * (C_tilde @ x).real)(xs)
    else:
        return xs[jnp.newaxis, -1], jax.vmap(lambda x: (C_tilde @ x).real)(xs)


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

    def setup(self):
        # In conjugate-symmetric parameterization, the state dimension remains P (complex),
        # we keep local_P = P. Real/imag handling is done by packing/unpacking, and outputs
        # use 2 * Re(C @ x) when conj_sym=True.
        # if self.conj_sym:
        #     local_P = 2 * self.P
        # else:
        #     local_P = self.P
        local_P = self.P

        self.Lambda_re = self.param("Lambda_re", lambda rng, shape: self.Lambda_re_init, (None,))
        self.Lambda_im = self.param("Lambda_im", lambda rng, shape: self.Lambda_im_init, (None,))
        if self.clip_eigs:
            self.Lambda = jnp.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im
        else:
            self.Lambda = self.Lambda_re + 1j * self.Lambda_im

        B_init = lecun_normal()
        B_shape = (local_P, self.H)
        self.B = self.param(
            "B",
            lambda rng, shape: init_VinvB(B_init, rng, shape, self.Vinv),
            B_shape,
        )
        B_tilde = self.B[..., 0] + 1j * self.B[..., 1]

        if self.C_init in ["trunc_standard_normal"]:
            C_init = trunc_standard_normal
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["lecun_normal"]:
            C_init = lecun_normal()
            C_shape = (self.H, local_P, 2)
        elif self.C_init in ["complex_normal"]:
            C_init = normal(stddev=0.5 ** 0.5)
        else:
            raise NotImplementedError(
                f"C_init method {self.C_init} not implemented"
            )

        if self.C_init in ["complex_normal"]:
            if self.bidirectional:
                C = self.param("C", C_init, (self.H, 2 * self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]
            else:
                C = self.param("C", C_init, (self.H, self.P, 2))
                self.C_tilde = C[..., 0] + 1j * C[..., 1]
        else:
            if self.bidirectional:
                self.C1 = self.param(
                    "C1", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape
                )
                self.C2 = self.param(
                    "C2", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape
                )
                C1 = self.C1[..., 0] + 1j * self.C1[..., 1]
                C2 = self.C2[..., 0] + 1j * self.C2[..., 1]
                self.C_tilde = jnp.concatenate((C1, C2), axis=-1)
            else:
                self.C = self.param(
                    "C", lambda rng, shape: init_CV(C_init, rng, shape, self.V), C_shape
                )
                self.C_tilde = self.C[..., 0] + 1j * self.C[..., 1]

        self.D = self.param("D", normal(stddev=1.0), (self.H,))

        self.log_step = self.param(
            "log_step",
            init_log_steps,
            (self.P, self.dt_min, self.dt_max),
        )
        step = self.step_rescale * jnp.exp(self.log_step[:, 0])

        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(self.Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(self.Lambda, B_tilde, step)
        else:
            raise NotImplementedError(
                f"Discretization method {self.discretization} not implemented"
            )

    def __call__(self, hidden, input_sequence, resets):
        hidden, ys = apply_ssm(
            self.Lambda_bar,
            self.B_bar,
            self.C_tilde,
            hidden,
            input_sequence,
            resets,
            self.conj_sym,
            self.bidirectional,
        )
        Du = jax.vmap(lambda u: self.D * u)(input_sequence)
        return hidden, ys + Du


def init_S5SSM(
    H,
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
    bidirectional,
):
    return partial(
        S5SSM,
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
        bidirectional=bidirectional,
    )


def make_HiPPO(N):
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, None] * P[None, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    return -A


def make_NPLR_HiPPO(N):
    hippo = make_HiPPO(N)
    P = jnp.sqrt(jnp.arange(N) + 0.5)
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    return hippo, P, B


def make_DPLR_HiPPO(N):
    A, P, B = make_NPLR_HiPPO(N)
    S = A + P[:, None] * P[None, :]
    S_diag = jnp.diagonal(S)
    Lambda_real = jnp.mean(S_diag) * jnp.ones_like(S_diag)
    Lambda_imag, V = eigh(S * -1j)
    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig


class SequenceLayer(nn.Module):
    ssm: partial[S5SSM] | type[S5SSM]
    d_model: int
    activation: str = "gelu"
    do_norm: bool = True
    prenorm: bool = True
    do_gtrxl_norm: bool = True
    step_rescale: float = 1.0

    def setup(self):
        self.seq = self.ssm(step_rescale=self.step_rescale) # type: ignore
        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)
        self.norm = nn.LayerNorm()

    def __call__(self, hidden, x, d):
        skip = x
        if self.prenorm and self.do_norm:
            x = self.norm(x)
        hidden, x = jax.vmap(lambda h, xx, dd: self.seq(h, xx, dd), in_axes=1, out_axes=1)(hidden, x, d)
        if self.do_gtrxl_norm:
            x = self.norm(x)
        if self.activation in ["full_glu"]:
            x = nn.gelu(x)
            x = self.out1(x) * jax.nn.sigmoid(self.out2(x))
        elif self.activation in ["half_glu1"]:
            x = nn.gelu(x)
            x = x * jax.nn.sigmoid(self.out2(x))
        elif self.activation in ["half_glu2"]:
            x1 = nn.gelu(x)
            x = x * jax.nn.sigmoid(self.out2(x1))
        elif self.activation in ["gelu"]:
            x = nn.gelu(x)
        else:
            raise NotImplementedError(
                f"Activation: {self.activation} not implemented"
            )
        x = skip + x
        if not self.prenorm and self.do_norm:
            x = self.norm(x)
        return hidden, x

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((1, batch_size, hidden_size), dtype=jnp.complex64)


class StackedEncoderModel(nn.Module):
    ssm: partial[S5SSM] | type[S5SSM]
    d_model: int
    n_layers: int
    activation: str = "gelu"
    do_norm: bool = True
    prenorm: bool = True
    do_gtrxl_norm: bool = True

    def setup(self):
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
        new_hiddens = []
        for i, layer in enumerate(self.layers):
            new_h, x = layer(hidden[i], x, d)
            new_hiddens.append(new_h)
        return new_hiddens, x

    @staticmethod
    def initialize_carry(batch_size, hidden_size, n_layers):
        return [jnp.zeros((1, batch_size, hidden_size), dtype=jnp.complex64) for _ in range(n_layers)]

