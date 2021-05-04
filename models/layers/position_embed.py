from typing import Callable

from flax.linen import initializers
from flax import linen as nn


class AddAbsPosEmbed(nn.Module):

    embed_init: Callable = initializers.normal(stddev=0.02)

    @nn.compact
    def __call__(self, inputs):
        pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
        pos_emb = self.param('pos_embed', self.embed_init, pos_emb_shape)
        output = inputs + pos_emb
        return output
    
class RotaryEmbed(nn.Module):
    def fixed_pos_embedding(x, seq_dim=0):
        dim = x.shape[-1]
        inv_freq = 1. / (10000 ** (np.arange(0, dim, 2) / dim))

        sinusoid_inp = np.einsum('i , j -> i j', np.arange(x.shape[seq_dim]), inv_freq)
    
        return np.sin(sinusoid_inp), np.cos(sinusoid_inp)


    def rotate_every_two(x):
        x1 = x[:, :, ::2]
        x2 = x[:, :, 1::2]

        x = jnp.stack((-x2, x1), axis=-1)

        return rearrange(x, '... d j -> ... (d j)')


    def apply_rotary_pos_emb(x, sincos):
        sin, cos = map(lambda t: repeat(t, 'b n -> b (n j)', j=2)[:, None, :], sincos)
        return (x * cos) + (rotate_every_two(x) * sin)
