"""A generic transformer network for policies and proxies.

See https://docs.kidger.site/equinox/examples/bert/ for the source.
"""

import chex
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int


class EmbedderBlock(eqx.Module):
    """Transformer embedder."""

    token_embedder: eqx.nn.Embedding
    position_embedder: eqx.nn.Embedding
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_size: int,
        hidden_size: int,
        dropout_rate: float,
        *,
        key: chex.PRNGKey,
    ):
        token_key, position_key = jax.random.split(key)

        self.token_embedder = eqx.nn.Embedding(
            num_embeddings=vocab_size,
            embedding_size=embedding_size,
            key=token_key,
        )
        self.position_embedder = eqx.nn.Embedding(
            num_embeddings=max_length,
            embedding_size=embedding_size,
            key=position_key,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        token_ids: Int[Array, " seq_len"],
        position_ids: Int[Array, " seq_len"],
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        tokens = jax.vmap(self.token_embedder)(token_ids)
        positions = jax.vmap(self.position_embedder)(position_ids)
        embedded_inputs = tokens + positions
        embedded_inputs = jax.vmap(self.layernorm)(embedded_inputs)
        embedded_inputs = self.dropout(
            embedded_inputs, inference=not enable_dropout, key=key
        )
        return embedded_inputs


class FeedForwardBlock(eqx.Module):
    """A single transformer feed forward block."""

    mlp: eqx.nn.Linear
    output: eqx.nn.Linear
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        dropout_rate: float,
        *,
        key: chex.PRNGKey,
    ):
        mlp_key, output_key = jax.random.split(key)
        self.mlp = eqx.nn.Linear(
            in_features=hidden_size,
            out_features=intermediate_size,
            key=mlp_key,
        )
        self.output = eqx.nn.Linear(
            in_features=intermediate_size,
            out_features=hidden_size,
            key=output_key,
        )

        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, " hidden_size"],
        enable_dropout: bool = True,
        key: chex.PRNGKey | None = None,
    ) -> Float[Array, " hidden_size"]:
        # Feed-forward.
        hidden = self.mlp(inputs)
        hidden = jax.nn.gelu(hidden)

        # Project back to input size.
        output = self.output(hidden)
        output = self.dropout(output, inference=not enable_dropout, key=key)

        # Residual and layer norm.
        output += inputs
        output = self.layernorm(output)

        return output


class AttentionBlock(eqx.Module):
    """A single transformer attention block."""

    attention: eqx.nn.MultiheadAttention
    layernorm: eqx.nn.LayerNorm
    dropout: eqx.nn.Dropout
    num_heads: int = eqx.field(static=True)

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        *,
        key: chex.PRNGKey,
    ):
        self.num_heads = num_heads
        self.attention = eqx.nn.MultiheadAttention(
            num_heads=num_heads,
            query_size=hidden_size,
            use_query_bias=True,
            use_key_bias=True,
            use_value_bias=True,
            use_output_bias=True,
            dropout_p=attention_dropout_rate,
            key=key,
        )
        self.layernorm = eqx.nn.LayerNorm(shape=hidden_size)
        self.dropout = eqx.nn.Dropout(dropout_rate)

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        mask: Int[Array, " seq_len"] | None,
        enable_dropout: bool = False,
        key: chex.PRNGKey = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        if mask is not None:
            mask = self.make_self_attention_mask(mask)
        attention_key, dropout_key = (
            (None, None) if key is None else jax.random.split(key)
        )

        attention_output = self.attention(
            query=inputs,
            key_=inputs,
            value=inputs,
            mask=mask,
            inference=not enable_dropout,
            key=attention_key,
        )

        result = attention_output
        result = self.dropout(result, inference=not enable_dropout, key=dropout_key)
        result = result + inputs
        result = jax.vmap(self.layernorm)(result)
        return result

    def make_self_attention_mask(
        self, mask: Int[Array, " seq_len"]
    ) -> Float[Array, "num_heads seq_len seq_len"]:
        """Create self-attention mask from sequence-level mask."""
        mask = jnp.multiply(
            jnp.expand_dims(mask, axis=-1), jnp.expand_dims(mask, axis=-2)
        )
        mask = jnp.expand_dims(mask, axis=-3)
        mask = jnp.repeat(mask, repeats=self.num_heads, axis=-3)
        return mask.astype(jnp.float32)


class TransformerLayer(eqx.Module):
    """A single transformer layer."""

    attention_block: AttentionBlock
    ff_block: FeedForwardBlock

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float,
        *,
        key: chex.PRNGKey,
    ):
        attention_key, ff_key = jax.random.split(key)

        self.attention_block = AttentionBlock(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            key=attention_key,
        )
        self.ff_block = FeedForwardBlock(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dropout_rate=dropout_rate,
            key=ff_key,
        )

    def __call__(
        self,
        inputs: Float[Array, "seq_len hidden_size"],
        mask: Int[Array, " seq_len"] | None = None,
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> Float[Array, "seq_len hidden_size"]:
        attn_key, ff_key = (None, None) if key is None else jax.random.split(key)
        attention_output = self.attention_block(
            inputs, mask, enable_dropout=enable_dropout, key=attn_key
        )
        seq_len = inputs.shape[0]
        ff_keys = None if ff_key is None else jax.random.split(ff_key, num=seq_len)
        output = jax.vmap(self.ff_block, in_axes=(0, None, 0))(
            attention_output, enable_dropout, ff_keys
        )
        return output


class Encoder(eqx.Module):
    """Full transformer encoder."""

    embedder_block: EmbedderBlock
    layers: list[TransformerLayer]
    pad_id: int

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        embedding_size: int,
        hidden_size: int,
        intermediate_size: int,
        num_layers: int,
        num_heads: int,
        dropout_rate: float,
        attention_dropout_rate: float = 0.0,
        pad_id: int = 0,
        *,
        key: chex.PRNGKey,
    ):
        self.pad_id = pad_id  # Padding token to identify masks
        embedder_key, layer_key = jax.random.split(key, num=2)
        self.embedder_block = EmbedderBlock(
            vocab_size=vocab_size,
            max_length=max_length,
            embedding_size=embedding_size,
            hidden_size=hidden_size,
            dropout_rate=dropout_rate,
            key=embedder_key,
        )

        layer_keys = jax.random.split(layer_key, num=num_layers)
        self.layers = []
        for layer_key in layer_keys:
            self.layers.append(
                TransformerLayer(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_heads=num_heads,
                    dropout_rate=dropout_rate,
                    attention_dropout_rate=attention_dropout_rate,
                    key=layer_key,
                )
            )

    def __call__(
        self,
        token_ids: Int[Array, " seq_len"],
        position_ids: Int[Array, " seq_len"],
        *,
        enable_dropout: bool = False,
        key: chex.PRNGKey | None = None,
    ) -> dict[str, Array]:
        emb_key, l_key = (None, None) if key is None else jax.random.split(key)

        embeddings = self.embedder_block(
            token_ids=token_ids,
            position_ids=position_ids,
            enable_dropout=enable_dropout,
            key=emb_key,
        )

        # We assume that all pad_id values should be masked out.
        mask = jnp.asarray(token_ids != self.pad_id, dtype=jnp.int32)

        x = embeddings
        layer_outputs = []
        for layer in self.layers:
            cl_key, l_key = (None, None) if l_key is None else jax.random.split(l_key)
            x = layer(x, mask, enable_dropout=enable_dropout, key=cl_key)
            layer_outputs.append(x)

        return {"embeddings": embeddings, "layers_out": layer_outputs}
