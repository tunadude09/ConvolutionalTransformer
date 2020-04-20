# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformer model from "Attention Is All You Need".

The Transformer model consists of an encoder and a decoder. Both are stacks
of self-attention layers followed by feed-forward layers. This model yields
good results on a number of problems, especially in NLP and machine translation.

See "Attention Is All You Need" (https://arxiv.org/abs/1706.03762) for the full
description of the model and the results obtained with its early version.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from six.moves import range  # pylint: disable=redefined-builtin

from tensor2tensor.data_generators import librispeech
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.layers import transformer_layers
from tensor2tensor.layers import transformer_memory
from tensor2tensor.utils import beam_search
from tensor2tensor.utils import expert_utils
from tensor2tensor.utils import mlperf_log
from tensor2tensor.utils import registry
from tensor2tensor.utils import t2t_model

import tensorflow as tf
import numpy as np

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops import inplace_ops
from tensorflow.python.util import nest
# pylint: enable=g-direct-tensorflow-import


#from tensorflow.python.keras.layers.convolutional import Conv1D
#from keras.layers import Conv1D
#import keras











"""
common_layers.conv_relu_conv
"""
#  common_attention.multihead_attention
#
#
'''
def conv_relu_conv(inputs,
                   filter_size,
                   output_size,
                   first_kernel_size=3,
                   second_kernel_size=3,
                   padding="SAME",
                   nonpadding_mask=None,
                   dropout=0.0,
                   name=None,
                   cache=None,
                   decode_loop_step=None):
  """Hidden layer with RELU activation followed by linear projection.
  Args:
    inputs: A tensor.
    filter_size: An integer.
    output_size: An integer.
    first_kernel_size: An integer.
    second_kernel_size: An integer.
    padding: A string.
    nonpadding_mask: A tensor.
    dropout: A float.
    name: A string.
    cache: A dict, containing Tensors which are the results of previous
        attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop.
        Only used for inference on TPU. If it is not None, the function
        will do inplace update for the cache instead of concatenating the
        current result to the cache.
  Returns:
    A Tensor.
  """
  with tf.variable_scope(name, "conv_relu_conv", [inputs]):
    inputs = maybe_zero_out_padding(inputs, first_kernel_size, nonpadding_mask)

    if cache:
      if decode_loop_step is None:
        inputs = cache["f"] = tf.concat([cache["f"], inputs], axis=1)
      else:
        # Inplace update is required for inference on TPU.
        # Inplace_ops only supports inplace_update on the first dimension.
        # The performance of current implementation is better than updating
        # the tensor by adding the result of matmul(one_hot,
        # update_in_current_step)
        tmp_f = tf.transpose(cache["f"], perm=[1, 0, 2])
        tmp_f = inplace_ops.alias_inplace_update(
            tmp_f,
            decode_loop_step * tf.shape(inputs)[1],
            tf.transpose(inputs, perm=[1, 0, 2]))
        inputs = cache["f"] = tf.transpose(tmp_f, perm=[1, 0, 2])
      inputs = cache["f"] = inputs[:, -first_kernel_size:, :]

    h = tpu_conv1d(
        inputs, filter_size, first_kernel_size, padding=padding, name="conv1")

    if cache:
      h = h[:, -1:, :]

    h = tf.nn.relu(h)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    h = maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
    return tpu_conv1d(
        h, output_size, second_kernel_size, padding=padding, name="conv2")








def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        attention_type="dot_product",
                        max_relative_position=None,
                        heads_share_relative_embedding=False,
                        add_relative_to_values=False,
                        image_shapes=None,
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name="multihead_attention",
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        vars_3d=False,
                        layer_collection=None,
                        recurrent_memory=None,
                        chunk_number=None,
                        hard_attention_k=0,
                        max_area_width=1,
                        max_area_height=1,
                        memory_height=1,
                        area_key_mode="mean",
                        area_value_mode="sum",
                        training=True,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    heads_share_relative_embedding: boolean to share relative embeddings
    add_relative_to_values: a boolean for whether to add relative component to
                            values.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    vars_3d: use 3-dimensional variables for input/output transformations
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
    recurrent_memory: An optional transformer_memory.RecurrentMemory, which
      retains state across chunks. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k).
    max_area_width: the max width allowed for an area.
    max_area_height: the max height allowed for an area.
    memory_height: the height of the memory.
    area_key_mode: the mode for computing area keys, which can be "mean",
      "concat", "sum", "sample_concat", and "sample_sum".
    area_value_mode: the mode for computing area values, which can be either
      "mean", or "sum".
    training: indicating if it is in the training mode.
    **kwargs (dict): Parameters for the attention function.
  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.
    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hidden_dim] rather than the full memory.
  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  vars_3d_num_heads = num_heads if vars_3d else 0

  if layer_collection is not None:
    if cache is not None:
      raise ValueError("KFAC implementation only supports cache is None.")
    if vars_3d:
      raise ValueError("KFAC implementation does not support 3d vars.")

  if recurrent_memory is not None:
    if memory_antecedent is not None:
      raise ValueError("Recurrent memory requires memory_antecedent is None.")
    if cache is not None:
      raise ValueError("Cache is not supported when using recurrent memory.")
    if vars_3d:
      raise ValueError("3d vars are not supported when using recurrent memory.")
    if layer_collection is not None:
      raise ValueError("KFAC is not supported when using recurrent memory.")
    if chunk_number is None:
      raise ValueError("chunk_number is required when using recurrent memory.")

  with tf.variable_scope(name, default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):

    if recurrent_memory is not None:
      (
          recurrent_memory_transaction,
          query_antecedent, memory_antecedent, bias,
      ) = recurrent_memory.pre_attention(
          chunk_number,
          query_antecedent, memory_antecedent, bias,
      )

    if cache is None or memory_antecedent is None:
      q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                            total_key_depth, total_value_depth, q_filter_width,
                            kv_filter_width, q_padding, kv_padding,
                            vars_3d_num_heads=vars_3d_num_heads,
                            layer_collection=layer_collection)
    if cache is not None:
      if attention_type not in ["dot_product", "dot_product_relative"]:
        # TODO(petershaw): Support caching when using relative position
        # representations, i.e. "dot_product_relative" attention.
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")

      if memory_antecedent is not None:
        # Encoder-Decoder Attention Cache
        q = compute_attention_component(query_antecedent, total_key_depth,
                                        q_filter_width, q_padding, "q",
                                        vars_3d_num_heads=vars_3d_num_heads)
        k = cache["k_encdec"]
        v = cache["v_encdec"]
      else:
        k = split_heads(k, num_heads)
        v = split_heads(v, num_heads)
        decode_loop_step = kwargs.get("decode_loop_step")
        if decode_loop_step is None:
          k = cache["k"] = tf.concat([cache["k"], k], axis=2)
          v = cache["v"] = tf.concat([cache["v"], v], axis=2)
        else:
          # Inplace update is required for inference on TPU.
          # Inplace_ops only supports inplace_update on the first dimension.
          # The performance of current implementation is better than updating
          # the tensor by adding the result of matmul(one_hot,
          # update_in_current_step)
          tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
          tmp_k = inplace_ops.alias_inplace_update(
              tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
          k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
          tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
          tmp_v = inplace_ops.alias_inplace_update(
              tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
          v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

    q = split_heads(q, num_heads)
    if cache is None:
      k = split_heads(k, num_heads)
      v = split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    if not vars_3d:
      q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      if max_area_width > 1 or max_area_height > 1:
        x = area_attention.dot_product_area_attention(
            q, k, v, bias, dropout_rate, image_shapes,
            save_weights_to=save_weights_to,
            dropout_broadcast_dims=dropout_broadcast_dims,
            max_area_width=max_area_width,
            max_area_height=max_area_height,
            memory_height=memory_height,
            area_key_mode=area_key_mode,
            area_value_mode=area_value_mode,
            training=training)
      else:
        x = dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
                                  save_weights_to=save_weights_to,
                                  make_image_summary=make_image_summary,
                                  dropout_broadcast_dims=dropout_broadcast_dims,
                                  activation_dtype=kwargs.get(
                                      "activation_dtype"),
                                  hard_attention_k=hard_attention_k)
    elif attention_type == "dot_product_relative":
      x = dot_product_attention_relative(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          save_weights_to=save_weights_to,
          make_image_summary=make_image_summary,
          cache=cache is not None,
          allow_memory=recurrent_memory is not None,
          hard_attention_k=hard_attention_k)
    elif attention_type == "dot_product_unmasked_relative_v2":
      x = dot_product_unmasked_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values)
    elif attention_type == "dot_product_relative_v2":
      x = dot_product_self_attention_relative_v2(
          q,
          k,
          v,
          bias,
          max_relative_position,
          dropout_rate,
          image_shapes,
          make_image_summary=make_image_summary,
          dropout_broadcast_dims=dropout_broadcast_dims,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values)
    elif attention_type == "local_within_block_mask_right":
      x = masked_within_block_local_attention_1d(
          q, k, v, block_length=block_length)
    elif attention_type == "local_relative_mask_right":
      x = masked_relative_local_attention_1d(
          q,
          k,
          v,
          block_length=block_length,
          make_image_summary=make_image_summary,
          dropout_rate=dropout_rate,
          heads_share_relative_embedding=heads_share_relative_embedding,
          add_relative_to_values=add_relative_to_values,
          name="masked_relative_local_attention_1d")
    elif attention_type == "local_mask_right":
      x = masked_local_attention_1d(
          q,
          k,
          v,
          block_length=block_length,
          make_image_summary=make_image_summary)
    elif attention_type == "local_unmasked":
      x = local_attention_1d(
          q, k, v, block_length=block_length, filter_width=block_width)
    elif attention_type == "masked_dilated_1d":
      x = masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
                                           gap_size, num_memory_blocks)
    else:
      assert attention_type == "unmasked_dilated_1d"
      x = dilated_self_attention_1d(q, k, v, block_length, block_width,
                                    gap_size, num_memory_blocks)
    x = combine_heads(x)

    # Set last dim specifically.
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

    if vars_3d:
      o_var = tf.get_variable(
          "o", [num_heads, total_value_depth // num_heads, output_depth])
      o_var = tf.cast(o_var, x.dtype)
      o_var = tf.reshape(o_var, [total_value_depth, output_depth])
      x = tf.tensordot(x, o_var, axes=1)
    else:
      x = common_layers.dense(
          x, output_depth, use_bias=False, name="output_transform",
          layer_collection=layer_collection)

    if recurrent_memory is not None:
      x = recurrent_memory.post_attention(recurrent_memory_transaction, x)
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x


def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID",
                vars_3d_num_heads=0,
                layer_collection=None):
  """Computes query, key and value.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    vars_3d_num_heads: an optional (if we want to use 3d variables)
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  q = compute_attention_component(
      query_antecedent,
      total_key_depth,
      q_filter_width,
      q_padding,
      "q",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  k = compute_attention_component(
      memory_antecedent,
      total_key_depth,
      kv_filter_width,
      kv_padding,
      "k",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  v = compute_attention_component(
      memory_antecedent,
      total_value_depth,
      kv_filter_width,
      kv_padding,
      "v",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  return q, k, v

def compute_attention_component(antecedent,
                                total_depth,
                                filter_width=1,
                                padding="VALID",
                                name="c",
                                vars_3d_num_heads=0,
                                layer_collection=None):
  """Computes attention compoenent (query, key or value).
  Args:
    antecedent: a Tensor with shape [batch, length, channels]
    total_depth: an integer
    filter_width: An integer specifying how wide you want the attention
      component to be.
    padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    name: a string specifying scope name.
    vars_3d_num_heads: an optional integer (if we want to use 3d variables)
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
  Returns:
    c : [batch, length, depth] tensor
  """
  if layer_collection is not None:
    if filter_width != 1 or vars_3d_num_heads != 0:
      raise ValueError(
          "KFAC implementation only supports filter_width=1 (actual: {}) and "
          "vars_3d_num_heads=0 (actual: {}).".format(
              filter_width, vars_3d_num_heads))
  if vars_3d_num_heads > 0:
    assert filter_width == 1
    input_depth = antecedent.get_shape().as_list()[-1]
    depth_per_head = total_depth // vars_3d_num_heads
    initializer_stddev = input_depth ** -0.5
    if "q" in name:
      initializer_stddev *= depth_per_head ** -0.5
    var = tf.get_variable(
        name, [input_depth,
               vars_3d_num_heads,
               total_depth // vars_3d_num_heads],
        initializer=tf.random_normal_initializer(stddev=initializer_stddev))
    var = tf.cast(var, antecedent.dtype)
    var = tf.reshape(var, [input_depth, total_depth])
    return tf.tensordot(antecedent, var, axes=1)
  if filter_width == 1:
    return common_layers.dense(
        antecedent, total_depth, use_bias=False, name=name,
        layer_collection=layer_collection)
  else:
    return common_layers.conv1d(
        antecedent, total_depth, filter_width, padding=padding, name=name)

def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True,
                          save_weights_to=None,
                          dropout_broadcast_dims=None,
                          activation_dtype=None,
                          weight_dtype=None,
                          hard_attention_k=0):
  """Dot-product attention.
  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims: an optional list of integers less than rank of q.
      Specifies in which dimensions to broadcast the dropout decisions.
    activation_dtype: Used to define function activation dtype when using
      mixed precision.
    weight_dtype: The dtype weights are stored in when using mixed precision
    hard_attention_k: integer, if > 0 triggers hard attention (picking top-k)
  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    if bias is not None:
      bias = common_layers.cast_like(bias, logits)
      logits += bias
    # If logits are fp16, upcast before softmax
    logits = maybe_upcast(logits, activation_dtype, weight_dtype)
    weights = tf.nn.softmax(logits, name="attention_weights")
    if hard_attention_k > 0:
      weights = harden_attention_weights(weights, hard_attention_k)
    weights = common_layers.cast_like(weights, q)
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = logits
    # Drop out attention links for each head.
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and make_image_summary:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)



'''

# Alias some commonly reused layers, here and elsewhere.
# transformer_prepare_encoder = transformer_layers.transformer_prepare_encoder
# transformer_encoder = transformer_layers.transformer_encoder
# transformer_ffn_layer = transformer_layers.transformer_ffn_layer










































"""
**************************************************************************************************************************
**************************************************************************************************************************
TODO: Put Conv Transformer Layer here!  FIXME:(Might need to replace these with updated ones form the T2T library)
**************************************************************************************************************************
**************************************************************************************************************************
"""


# def dilated_causal_conv1d(inputs, filters, kernel_size=3, strides=1,
#                      dilation_rate=1, padding="CAUSAL"):
#   if padding.lower() == "causal":
#     #  TODO: I need dilated, causal 1d convolutions here
#     channel_size = inputs.get_shape()[-1]
#
#     padding = (kernel_size - 1) * dilation_rate
#     inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
#     # ts = tf.TensorShape([tf.Dimension(None),tf.Dimension(None),tf.Dimension(channel_size)])
#     inputs.set_shape([None,None,channel_size])
#     # return super(CausalConv1D, self).call(inputs)
#
#     # return Conv1D(filters, kernel_size=kernel_size,
#     #               strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
#
#
#     # return tf.layers.conv1d(inputs, filters, kernel_size=kernel_size,
#     #               strides=strides, padding="VALID", dilation_rate=dilation_rate)
#     return keras.layers.Conv1D(filters, kernel_size=kernel_size,
#                   strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
#
#   else:
#     # return Conv1D(filters, kernel_size=kernel_size,
#     #               strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
#     # return tf.layers.conv1d(inputs, filters, kernel_size=kernel_size,
#     #               strides=strides, padding=padding, dilation_rate=dilation_rate)
#     return keras.layers.Conv1D(filters, kernel_size=kernel_size,
#                   strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
#
#
#
# def dilated_causal_conv2d(inputs, filters, kernel_size=3, strides=1,
#                      dilation_rate=1, padding="CAUSAL", num_heads=1):
#   if padding.lower() == "causal":
#     #  TODO: I need dilated, causal 2d convolutions here
#     channel_size = inputs.get_shape()[-1]
#
#     padding = (kernel_size - 1) * dilation_rate
#     inputs = tf.pad(inputs, tf.constant([(0, 0,), (0, 0,), (1, 0), (0, 0)]) * padding)
#     # ts = tf.TensorShape([tf.Dimension(None),tf.Dimension(None),tf.Dimension(channel_size)])
#     inputs.set_shape([None,num_heads,None,channel_size])
#
#     # return tf.layers.conv2d(inputs, filters, kernel_size=kernel_size,
#     #               strides=strides, padding="VALID", dilation_rate=dilation_rate)
#     return keras.layers.Conv2D(filters, kernel_size=kernel_size,
#                   strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
#
#   else:
#
#     # return tf.layers.conv2d(inputs, filters, kernel_size=kernel_size,
#     #               strides=strides, padding=padding, dilation_rate=dilation_rate)
#     return keras.layers.Conv2D(filters, kernel_size=kernel_size,
#                   strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)























###################################################################################
###################################################################################
##################################################################################
###################################################################################
###################################################################################
##################################################################################
###################################################################################
###################################################################################
##################################################################################
###################################################################################
###################################################################################
##################################################################################


#def dilated_causal_conv2d(inputs, filters, kernel_size=3, strides=1,
#                     dilation_rate=1, padding="CAUSAL", num_heads=1):
#  if padding.lower() != "causal":
#    print("\n\n\n CAUSAL dilated_causal_conv2d \n\n\n\n")
#  else:
#    print("\n\n\n NOT CAUSAL dilated_causal_conv2d \n\n\n")
#  #print ("\n\n should be causal WARNING WARNING \n\n\n\n WARNING THIS SHOULD NOT BE CAUSEL IT'S HARDCODED BECAUSE OF \n\n  DUMB CONDITION TF STUFF  \n\n\n\n WARNING WARNING WARNING  \n\n\n  WARNING WARNING WARNING  WARNING")
#
#  #  TODO: I need dilated, causal 2d convolutions here
#  channel_size = inputs.get_shape()[-1]
#
#  padding = (kernel_size - 1) * dilation_rate
#  inputs = tf.pad(inputs, tf.constant([(0, 0,), (0, 0,), (1, 0), (0, 0)]) * padding)
#  # ts = tf.TensorShape([tf.Dimension(None),tf.Dimension(None),tf.Dimension(channel_size)])
#  inputs.set_shape([None,num_heads,None,channel_size])
#
##    # return tf.layers.conv2d(inputs, filters, kernel_size=kernel_size,
##    #               strides=strides, padding="VALID", dilation_rate=dilation_rate)
#  return keras.layers.Conv2D(filters, kernel_size=kernel_size,
#                  strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
#
##  else:
#
#  # return tf.layers.conv2d(inputs, filters, kernel_size=kernel_size,
#  #               strides=strides, padding=padding, dilation_rate=dilation_rate)
#  #return keras.layers.Conv2D(filters, kernel_size=kernel_size,
#  #                 strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
#








#def init_f(shape, dtype=None):
#        ker = np.zeros(shape, dtype=dtype)
#        ker[tuple(map(lambda x: int(np.floor(x/2)), ker.shape))]=1
#        return ker
#





#def conv1d_with_causal_padding(inputs, filters, kernel_size=3, strides=1,
#                     dilation_rate=1, padding="VALID", depthwise_sep=False):
#     #  TODO: I need dilated, causal 1d convolutions here
#     channel_size = inputs.get_shape()[-1]
#
#     padding = (kernel_size - 1) * dilation_rate
#     inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
#     # ts = tf.TensorShape([tf.Dimension(None),tf.Dimension(None),tf.Dimension(channel_size)])
#     inputs.set_shape([None,None,channel_size])
#     # return super(CausalConv1D, self).call(inputs)
#
#     # return Conv1D(filters, kernel_size=kernel_size,
#     #               strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
#
#
#     init_val = lambda shape, dtype=None: init_f(shape=shape, dtype=dtype)
#     return keras.layers.Conv1D(filters, kernel_size=kernel_size,
#                               strides=strides, padding="VALID", dilation_rate=dilation_rate,
#                                                 kernel_initializer=init_val, bias_initializer='zeros', kernel_regularizer=None)(inputs)
#

#def conv1d_keras(inputs, filters, kernel_size=3, strides=1,
#                                  dilation_rate=1, padding="VALID", depthwise_sep=False):
#    init_val = lambda shape, dtype=None: init_f(shape=shape, dtype=dtype)
#    return keras.layers.Conv1D(filters, kernel_size=kernel_size,
#                                       strides=strides, padding="VALID", dilation_rate=dilation_rate,
#                                                         kernel_initializer=init_val, bias_initializer='zeros', kernel_regularizer=None)(inputs)





#def depthwise_sep_dilated_causal_conv1d(inputs, filters, kernel_size=3, strides=1,
#                     dilation_rate=1, padding="CAUSAL", depthwise_sep=False):
#  if padding == "CAUSAL":
#    print("\n\n\n CAUSAL depthwise_sep_dilated_causal_conv1d \n\n\n")
#  else:
#    print("\n\n\n NOT CAUSAL depthwise_sep_dilated_cuasla_conv1d \n\n\n")
#  #print("BROKEN  BROKEN RROKEN BROKEN  \n\n\n  BROKEN NO CAUSAL PADDING BROKEN \n\n\n  BROKEN \n\n\n  BROKEN BROKEN OOFJOASIDUFAOSIUFOAISDUFOASIUFAOSIUFAOIUDFOIUSIAUDOFIASUFOSAUFIOUSADFOIUADSOIFUAOF BROKEN BROKEN BROKEN BROKEN BROKEN LKAJDLJASLFKJASLFKJASDLKFJASLFJASLDKFJLSKDJ BROKEN BROKEN BROKEN")
#  return conv1d_keras(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
#          dilation_rate=dilation_rate, padding=padding, depthwise_sep=False)
#  #return tf.cond( tf.cast(padding.lower() == "causal", tf.bool),
#  #return tf.keras.backend.switch(padding.lower() == "causal",
#  #lambda: conv1d_with_causal_padding(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
#  #                     dilation_rate=dilation_rate, padding="VALID", depthwise_sep=False),
#  #lambda: onv1d_keras(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
#  #                     dilation_rate=dilation_rate, padding=padding, depthwise_sep=False))
#
#  #
#  # if padding.lower() == "causal":
#  #   return conv1d_with_causal_padding(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
#  #                        dilation_rate=dilation_rate, padding="VALID", depthwise_sep=False)
#  #
#  #
#  # else:
#  #
#  #     return conv1d_keras(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
#  #                          dilation_rate=dilation_rate, padding=padding, depthwise_sep=False)
#  #
#  #

###################################################################################
###################################################################################
##################################################################################
###################################################################################
###################################################################################
##################################################################################
###################################################################################
###################################################################################
##################################################################################
###################################################################################
###################################################################################
##################################################################################
###################################################################################
###################################################################################
##################################################################################






























def dilated_causal_conv2d(inputs, filters, kernel_size=3, strides=1,
                     dilation_rate=1, padding="CAUSAL", num_heads=1):
  if padding.lower() == "causal":
    #  TODO: I need dilated, causal 2d convolutions here
    channel_size = inputs.get_shape()[-1]

    padding="VALID"

    padding_rate = (kernel_size - 1) * dilation_rate
    inputs = tf.pad(inputs, tf.constant([(0, 0,), (0, 0,), (1, 0), (0, 0)]) * padding_rate)
    # ts = tf.TensorShape([tf.Dimension(None),tf.Dimension(None),tf.Dimension(channel_size)])
    inputs.set_shape([None,num_heads,None,channel_size])

    # return tf.layers.conv2d(inputs, filters, kernel_size=kernel_size,
    #               strides=strides, padding="VALID", dilation_rate=dilation_rate)
  return tf.keras.layers.Conv2D(filters, kernel_size=kernel_size,
                strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)











def depthwise_sep_dilated_causal_conv1d(inputs, filters, kernel_size=3, strides=1,
                     dilation_rate=1, padding="CAUSAL", depthwise_sep=False):
     channel_size = inputs.get_shape()[-1]

     if padding.lower() == "causal":
       #channel_size = inputs.shape[-1]

       #  TODO: I need dilated, causal 1d convolutions here

       padding="VALID"
       padding_rate = (kernel_size - 1) * dilation_rate
       inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding_rate)
       # ts = tf.TensorShape([tf.Dimension(None),tf.Dimension(None),tf.Dimension(channel_size)])
       inputs.set_shape([None,None,channel_size])


       # return super(CausalConv1D, self).call(inputs)

       # return Conv1D(filters, kernel_size=kernel_size,
       #               strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
     #return tf.layers.conv1d(filters, kernel_size=kernel_size,
     #             strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
     #return tf.keras.layers.Conv1D(filters, kernel_size=kernel_size,
#               strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)






     kernel_shape = [kernel_size, channel_size, filters]
     w = tf.get_variable(
          "DW", shape=kernel_shape)
     return tf.nn.conv1d(
          value=inputs,
          filters=w,
          stride=strides,
          padding=padding,
          dilations=dilation_rate,
          name=None)




#
#
# def depthwise_sep_dilated_causal_conv1d(inputs, filters, kernel_size=3, strides=1,
#                      dilation_rate=1, padding="CAUSAL", depthwise_sep=False):
#      if padding.lower() == "causal":
#
#        #  TODO: I need dilated, causal 1d convolutions here
#        channel_size = inputs.get_shape()[-1]
#
#        padding="VALID"
#        padding_rate = (kernel_size - 1) * dilation_rate
#        inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding_rate)
#        # ts = tf.TensorShape([tf.Dimension(None),tf.Dimension(None),tf.Dimension(channel_size)])
#        inputs.set_shape([None,None,channel_size])
#
#
#        # return super(CausalConv1D, self).call(inputs)
#
#        # return Conv1D(filters, kernel_size=kernel_size,
#        #               strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
#      #return tf.keras.layers.Conv1D(filters, kernel_size=kernel_size,
#      #              strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
#
#
#
#
#      kernel_shape = [kernel_size, inputs.get_shape()[-1], filters]
#      w = tf.get_variable(
#        "DW", shape=kernel_shape)
#      return tf.nn.conv1d(
#            inputs,
#            w,
#            strides,
#            padding,
#            dilations=dilation_rate,
#            name=None)
#
#




#
#
# def conv1d_without_causal_padding(inputs, filters, padding, kernel_size=3, strides=1,
#                      dilation_rate=1):
#     return keras.layers.Conv1D(filters, kernel_size=kernel_size,
#                   strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
#




#
#
#
#
# def conv1d_with_causal_padding(inputs, filters, kernel_size=3, strides=1,
#                      dilation_rate=1, padding="VALID"):
#      #  TODO: I need dilated, causal 1d convolutions here
#      channel_size = inputs.get_shape()[-1]
#
#      padding = (kernel_size - 1) * dilation_rate
#      inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
#      # ts = tf.TensorShape([tf.Dimension(None),tf.Dimension(None),tf.Dimension(channel_size)])
#      inputs.set_shape([None,None,channel_size])
#      # return super(CausalConv1D, self).call(inputs)
#
#      # return Conv1D(filters, kernel_size=kernel_size,
#      #               strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
#      return keras.layers.Conv1D(filters, kernel_size=kernel_size,
#                   strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
#
#
# def conv1d_without_causal_padding(inputs, filters, padding, kernel_size=3, strides=1,
#                      dilation_rate=1):
#     return keras.layers.Conv1D(filters, kernel_size=kernel_size,
#                   strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)


#
# def depthwise_sep_dilated_causal_conv1d(inputs, filters, kernel_size=3, strides=1,
#                      dilation_rate=1, padding="CAUSAL", depthwise_sep=False):
#   # return tf.cond( padding.lower() == "causal",
#   return tf.keras.backend.switch(padding.lower() == "causal",
#           conv1d_with_causal_padding(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
#                        dilation_rate=dilation_rate, padding="VALID", depthwise_sep=False),
#           conv1d_keras(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
#                        dilation_rate=dilation_rate, padding=padding, depthwise_sep=False))

  #
  # if padding.lower() == "causal":
  #   return conv1d_with_causal_padding(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
  #                        dilation_rate=dilation_rate, padding="VALID", depthwise_sep=False)
  #
  #
  # else:
  #
  #     return conv1d_keras(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides,
  #                          dilation_rate=dilation_rate, padding=padding, depthwise_sep=False)
  #
  #










########FIXME:  suspicious #################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
# def depthwise_sep_dilated_causal_conv1d(inputs, filters, kernel_size=3, strides=1,
#                      dilation_rate=1, padding="CAUSAL", depthwise_sep=False):
#   if padding.lower() == "causal":
#     #  TODO: I need dilated, causal 1d convolutions here
#     channel_size = inputs.get_shape()[-1]
#
#     padding = (kernel_size - 1) * dilation_rate
#     inputs = tf.pad(inputs, tf.constant([(0, 0,), (1, 0), (0, 0)]) * padding)
#     # ts = tf.TensorShape([tf.Dimension(None),tf.Dimension(None),tf.Dimension(channel_size)])
#     inputs.set_shape([None,None,channel_size])
#     # return super(CausalConv1D, self).call(inputs)
#
#     # return Conv1D(filters, kernel_size=kernel_size,
#     #               strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
#     if depthwise_sep:
#         return tf.keras.layers.SeparableConv1D(filters=filters,
#                       kernel_size=kernel_size,
#                       activation='relu',
#                       bias_initializer='random_uniform',
#                       depthwise_initializer='random_uniform',
#                       depth_multiplier=1,
#                       strides=strides,
#                       padding="VALID",
#                       dilation_rate=dilation_rate)(inputs)
#
#     else:
#
#
#
#
#         # return tf.layers.conv1d(inputs, filters, kernel_size=kernel_size,
#         #               strides=strides, padding="VALID", dilation_rate=dilation_rate)
#         return keras.layers.Conv1D(filters, kernel_size=kernel_size,
#                       strides=strides, padding="VALID", dilation_rate=dilation_rate)(inputs)
#
#   else:
#     # return Conv1D(filters, kernel_size=kernel_size,
#     #               strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
#       if depthwise_sep:
#           return tf.keras.layers.SeparableConv1D(filters=filters,
#                       kernel_size=kernel_size,
#                       activation='relu',
#                       bias_initializer='random_uniform',
#                       depthwise_initializer='random_uniform',
#                       depth_multiplier=1,
#                       strides=strides,
#                       padding=padding,
#                       dilation_rate=dilation_rate)(inputs)
#       else:
#           #  FIXME:  RESTART HERE!!! FIXME: FIXME: FIXME: that's awesome!
#           #  FIXME:
#           #  FIXME:
#
#           #return tf.layers.conv1d(inputs, filters, kernel_size=kernel_size,
#           #             strides=strides, padding=padding, dilation_rate=dilation_rate)
#
#           return keras.layers.Conv1D(filters, kernel_size=kernel_size,
#                        strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)
#
#           #return tf.nn.convolution(inputs, filters,
#           #            strides=strides, padding=padding, dilation_rate=dilation_rate)
#
#           #return Conv1D(filters, kernel_size=kernel_size,
#           #            strides=strides, padding=padding, dilation_rate=dilation_rate)(inputs)

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################





#########################################################################################
# def depthwise_sep_conv_keras(input, channels, kernel_size=3, depth_multiplier=1):
#     with tf.variable_scope(name="depthwise_sep_conv_keras") as scope:
#         return tf.keras.layers.SeparableConv1D(filters=channels,
#                       kernel_size=kernel_size,
#                       activation='relu',
#                       bias_initializer='random_uniform',
#                       depthwise_initializer='random_uniform',
#                       padding='same',
#                       depth_multiplier=depth_multiplier)(input)

























































#common_attention.dot_product_attention()


'''
def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True,
                          save_weights_to=None,
                          dropout_broadcast_dims=None):
  """Dot-product attention.

  Args:
    q: Tensor with shape [..., length_q, depth_k].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    bias: bias Tensor (see attention_bias())
    dropout_rate: a float.
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims: an optional list of integers less than rank of q.
      Specifies in which dimensions to broadcast the dropout decisions.

  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  with tf.variable_scope(
      name, default_name="dot_product_attention", values=[q, k, v]) as scope:
    logits = tf.matmul(q, k, transpose_b=True)  # [..., length_q, length_kv]
    if bias is not None:
      bias = common_layers.cast_like(bias, logits)
      logits += bias
    weights = tf.nn.softmax(logits, name="attention_weights")
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = logits
    # Drop out attention links for each head.
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and make_image_summary:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)'''



def conv_relu_conv(inputs,
                   filter_size,
                   output_size,
                   first_kernel_size=3,
                   second_kernel_size=3,
                   padding="SAME",
                   nonpadding_mask=None,
                   dropout=0.0,
                   name=None,
                   cache=None,
                   dil_rate1=1,
                   dil_rate2=1,
                   depthwise_sep=False,
                   decode_loop_step=None):
  """Hidden layer with RELU activation followed by linear projection."""
  with tf.variable_scope(name, "conv_relu_conv", [inputs]):
    inputs = common_layers.maybe_zero_out_padding(
        inputs, first_kernel_size, nonpadding_mask)

    if cache:
      if decode_loop_step is None:
        inputs = cache["f"] = tf.concat([cache["f"], inputs], axis=1)
      else:
        # Inplace update is required for inference on TPU.
        # Inplace_ops only supports inplace_update on the first dimension.
        # The performance of current implementation is better than updating
        # the tensor by adding the result of matmul(one_hot,
        # update_in_current_step)
        tmp_f = tf.transpose(cache["f"], perm=[1, 0, 2])
        tmp_f = inplace_ops.alias_inplace_update(
            tmp_f,
            decode_loop_step * tf.shape(inputs)[1],
            tf.transpose(inputs, perm=[1, 0, 2]))
        inputs = cache["f"] = tf.transpose(tmp_f, perm=[1, 0, 2])
      inputs = cache["f"] = inputs[:, -first_kernel_size:, :]

    #
    # #  FIXME: might be a hack, sometimes LEFT is being passed, set this to the default SAME
    # if padding not in {"SAME","VALID"}:
    #   padding = "SAME"
    if padding.lower() == "left":
      padding = "causal"

    # h = tpu_conv1d(inputs, filter_size, first_kernel_size, padding=padding,
    #                name="conv1")

    # W1 = tf.get_variable("ffn_conv_W1", [3,output_size,output_size],
    #                     initializer=tf.random_normal_initializer(stddev=0.001) )
    # h = tf.layers.conv1d(inputs, filters=filter_size, kernel_size=2, strides=1,
    #                      padding=padding, dilation_rate=1)



    #  TODO: I need dilated, causal 1d convolutions here
    # h = CausalConv1D(filters=filter_size, kernel_size=3, strides=1,
    #                   dilation_rate=1).call(inputs)


    h = depthwise_sep_dilated_causal_conv1d(inputs, filters=filter_size,
                              kernel_size=first_kernel_size, strides=1,
                              dilation_rate=dil_rate1, padding=padding,
                              depthwise_sep=depthwise_sep)
    # h = dilated_causal_conv1d(inputs, filters=filter_size,
    #                           kernel_size=first_kernel_size, strides=1,
    #                           dilation_rate=dil_rate1, padding=padding)


    # W1 = tf.get_variable("ffn_conv_W1", [3,3,output_size,output_size],
    #                     initializer=tf.random_normal_initializer(stddev=0.001) )
    # h = tf.nn.conv2d(inputs, W1, strides=[1,1,1,1],
    #                 dilations=[1,1,dil_rate1,dil_rate1], padding=padding,
    #                 name="conv1")


    if cache:
      h = h[:, -1:, :]

    h = tf.nn.relu(h)
    if dropout != 0.0:
      h = tf.nn.dropout(h, 1.0 - dropout)
    h = common_layers.maybe_zero_out_padding(h, second_kernel_size, nonpadding_mask)
    # return tpu_conv1d(h, output_size, second_kernel_size, padding=padding,
    #                   name="conv2")
    #  FIXME: using same padding switch back after
    # return tf.layers.conv1d(h, filters=output_size, kernel_size=2, strides=1,
    #                      padding="SAME", dilation_rate=1)


    return depthwise_sep_dilated_causal_conv1d(h, filters=output_size,
                                 kernel_size=second_kernel_size, strides=1,
                                 dilation_rate=dil_rate2, padding=padding,
                                 depthwise_sep=depthwise_sep)
    # return dilated_causal_conv1d(h, filters=output_size,
    #                              kernel_size=second_kernel_size, strides=1,
    #                              dilation_rate=dil_rate2, padding=padding)

    # # #  TODO:  fix this!
    # W2 = tf.get_variable("ffn_conv_W2", [3,3,output_size,output_size],
    #                     initializer=tf.random_normal_initializer(stddev=0.001) )
    # return tf.nn.conv2d(h, W2, strides=[1,1,1,1],
    #                     dilations=[1,1,dil_rate2,dil_rate2], padding=padding,
    #                     name="conv2")


def compute_attention_component(antecedent,
                                total_depth,
                                filter_width=1,
                                padding="VALID",
                                name="c",
                                vars_3d_num_heads=0,
                                dil_rate=1,
                                depthwise_sep=False,
                                layer_collection=None):
  """Computes attention compoenent (query, key or value).

  Args:
    antecedent: a Tensor with shape [batch, length, channels]
    total_depth: an integer
    filter_width: An integer specifying how wide you want the attention
      component to be.
    padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    name: a string specifying scope name.
    vars_3d_num_heads: an optional integer (if we want to use 3d variables)

  Returns:
    c : [batch, length, depth] tensor
  """
  if layer_collection is not None:
    if filter_width != 1 or vars_3d_num_heads != 0:
      raise ValueError(
          "KFAC implementation only supports filter_width=1 (actual: {}) and "
          "vars_3d_num_heads=0 (actual: {}).".format(
              filter_width, vars_3d_num_heads))
  if vars_3d_num_heads > 0:
    assert filter_width == 1
    input_depth = antecedent.get_shape().as_list()[-1]
    depth_per_head = total_depth // vars_3d_num_heads
    initializer_stddev = input_depth ** -0.5
    if "q" in name:
      initializer_stddev *= depth_per_head ** -0.5
    var = tf.get_variable(
        name, [input_depth,
               vars_3d_num_heads,
               total_depth // vars_3d_num_heads],
        initializer=tf.random_normal_initializer(stddev=initializer_stddev))
    var = tf.cast(var, antecedent.dtype)
    var = tf.reshape(var, [input_depth, total_depth])
    return tf.tensordot(antecedent, var, axes=1)
  if filter_width == 1:
    return common_layers.dense(
        antecedent, total_depth, use_bias=False, name=name,
        layer_collection=layer_collection)
  else:
    #return common_layers.conv1d(
    #      antecedent, total_depth, filter_width, padding=padding, name=name)







         ######################################################################
######################################################################
######################################################################
##############FIXME: quarentined for debug ########################################################
######################################################################
######################################################################
######################################################################

    #  FIXME:STEP#1
    #  TODO:  make sure the convolution works here
    # dilated_causal_conv1d(antecedent, total_depth, filter_width, padding)

    return depthwise_sep_dilated_causal_conv1d(antecedent, filters=total_depth,
                                 kernel_size=filter_width, strides=1,
                                 dilation_rate=dil_rate, padding=padding,
                                 depthwise_sep=depthwise_sep)

######################################################################
######################################################################
######################################################################
######################################################################
######################################################################
######################################################################











    # return dilated_causal_conv1d(antecedent, filters=total_depth,
    #                              kernel_size=filter_width, strides=1,
    #                              dilation_rate=dil_rate, padding=padding)


######################################################################
def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_value_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID",
                vars_3d_num_heads=0,
                v_padding="VALID",
                v_kernel_size=1,
                v_dil_rate=1,
                depthwise_sep=False,
                layer_collection=None):
  """Computes query, key and value.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels]
    total_key_depth: an integer
    total_value_depth: an integer
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
    to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    kv_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
    vars_3d_num_heads: an optional (if we want to use 3d variables)

  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    memory_antecedent = query_antecedent
  q = compute_attention_component(
      query_antecedent,
      total_key_depth,
      q_filter_width,
      q_padding,
      "q",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  k = compute_attention_component(
      memory_antecedent,
      total_key_depth,
      kv_filter_width,
      kv_padding,
      "k",
      vars_3d_num_heads=vars_3d_num_heads,
      layer_collection=layer_collection)
  #  FIXME:STEP#1:  I've changed the padding to CAUSAL
  v = compute_attention_component(
      memory_antecedent,
      total_value_depth,
      v_kernel_size,
      v_padding,
      "v",
      vars_3d_num_heads=vars_3d_num_heads,
      dil_rate=v_dil_rate,
      depthwise_sep=depthwise_sep,
      layer_collection=layer_collection)
  return q, k, v





#  TODO:
#  TODO:  this implementatio seems a little iffy, check
#  FIXME:STEP#1
def pointwise_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          make_image_summary=True,
                          save_weights_to=None,
                          dropout_broadcast_dims=None,
                          max_len=256,
                          padding="CAUSAL",
                          num_heads=1,
                          activation_dtype=None,
                          weight_dtype=None,
                          hard_attention_k=0):
  """dot-product attention.
  Args:
    q: a Tensor with shape [batch, heads, length_q, depth_k]
    k: a Tensor with shape [batch, heads, length_kv, depth_k]
    v: a Tensor with shape [batch, heads, length_kv, depth_v]
    bias: bias Tensor (see attention_bias())
    dropout_rate: a floating point number
    image_shapes: optional tuple of integer scalars.
      see comments for attention_image_summary()
    name: an optional string
    make_image_summary: True if you want an image summary.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
  Returns:
    A Tensor.
  """
  with tf.variable_scope(name, default_name="pointwise_attention", values=[q, k, v]) as scope:
    # [batch, num_heads, query_length, memory_length]
    #  FIXME:STEP#1
    #  TODO: I need to open tensorboard and confirm it is actually shaped like this


    # import pdb; pdb.set_trace()

    # q = tf.Print(q,[tf.shape(q)])
    # k = tf.Print(k,[tf.shape(k)])
    # v = tf.Print(v,[tf.shape(v)])


    logits_last_dim = k.get_shape()[-2]
    logits = tf.matmul(q, k, transpose_b=True)
    # logits = tf.Print(logits,[tf.shape(logits)])


    #  TODO:
    #  TODO:  this bias being added is the triangle mask, which zeros out all
    #  TODO:  forward information in the softmax function after, as described in
    #  TODO:  the paper
    if bias is not None:
      bias = common_layers.cast_like(bias, logits)
      logits += bias


    # import pdb; pdb.set_trace()

    # If logits are fp16, upcast before softmax
    logits = common_attention.maybe_upcast(logits, activation_dtype, weight_dtype)

    weights = tf.nn.softmax(logits, name="attention_weights")
    if save_weights_to is not None:
      save_weights_to[scope.name] = weights
      save_weights_to[scope.name + "/logits"] = logits























    #  FIXME:  this might be overkill, just trying to reduce dimensionality
    #  changing weights shape to match v, for element-wise multiplication
    #  averag every i values to match v's last dim size
    #  I thought about adding relu after this, but it's not a real layer since it's
    #  just 1x1 so I'll just leave it for now
    # pool_width = tf.floordiv(weights.get_shape()[-1], v.get_shape()[-1])
    # tf.nn.avg_pool(weights, [1,1,1,pool_width])
    # weights.set_shape([tf.shape(weights)[0],tf.shape(weights)[1],tf.shape(weights)[2],tf.shape(weights)[3]])

    # weights.set_shape([weights.get_shape()[0], weights.get_shape()[1],
                       # weights.get_shape()[2], seq_len])
    # weights.set_shape([weights.get_shape()[0], weights.get_shape()[1],
    #                   weights.get_shape()[2], logits_last_dim])


    #  TODO: FIXME:  change 64 to max_len / num_heads after testings
    # paddings = [[0, 0], [0, [0,64-tf.shape(weights)[-1]]]

    #  the padding is only used here to account for variable length sequneces
    #  it's just used to allow a conv2d to pre-configure weight matrixes for the
    #  max local size, not used anywhere else in the graph as shape is vital Here
    #  for the pointwise multiply operation at the end







    #  Need to fix last dimension
    #  TODO: FIXME:  FIXME:
    #  FIXME:  replace this padding
    paddings = [[0,0], [0,0], [0,0], [0,max_len-tf.shape(weights)[-1]]]
    weights = tf.pad(weights, paddings, 'CONSTANT', constant_values=-1)
    weights.set_shape([weights.get_shape()[0], weights.get_shape()[1],
                      weights.get_shape()[2], max_len])






    # FIXME:STEP#1
    # TODO: I only added this to change the shape, but I don't think it's Necessary
    # in vanilla transformer translation
    # weights = tf.layers.conv2d(weights, filters=v.get_shape()[-1], kernel_size=1,
    #                           strides=1, padding="SAME")
    # weights = dilated_causal_conv2d(weights, filters=v.get_shape()[-1], kernel_size=1,
    #                           strides=1, padding=padding, num_heads=num_heads)
    weights = dilated_causal_conv2d(weights, filters=v.get_shape().as_list()[-1], kernel_size=1,
                              strides=1, padding=padding, num_heads=num_heads)

























    # dropping out the attention links for each of the heads
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
    if common_layers.should_generate_summaries() and make_image_summary:
      common_attention.attention_image_summary(weights, image_shapes)
    #  FIXME:
    #  FIXME:
    #  FIXME: changed to multiply, will throw error if weights and v don't have
    #  FIXME: the same dimensions
    #  FIXME: looks like the actual attention operation is here, should replace with pointwise
    # return tf.matmul(weights, v)
    return tf.multiply(weights, v)



######################################################################
def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        attention_type="dot_product",
                        max_relative_position=None,
                        heads_share_relative_embedding=False,
                        add_relative_to_values=False,
                        image_shapes=None,
                        block_length=128,
                        block_width=128,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name="multihead_attention",
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        vars_3d=False,

                        q_filter_width=1,
                        kv_filter_width=1,

                        v_padding="VALID",
                        v_kernel_size=3,
                        v_dil_rate=1,
                        combine_dil_rate=1,
                        depthwise_sep=False,
                        max_length=256,


                        layer_collection=None,
                        recurrent_memory=None,
                        chunk_number=None,
                        hard_attention_k=0,
                        max_area_width=1,
                        max_area_height=1,
                        memory_height=1,
                        area_key_mode="mean",
                        area_value_mode="sum",
                        training=True,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.

  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d", graph, or any attention function
                    with the signature (query, key, value, **kwargs)
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    heads_share_relative_embedding: boolean to share relative embeddings
    add_relative_to_values: a boolean for whether to add relative component to
                            values.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string.
    save_weights_to: an optional dictionary to capture attention weights
      for vizualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    dropout_broadcast_dims:  an optional list of integers less than 4
      specifying in which dimensions to broadcast the dropout decisions.
      saves memory.
    vars_3d: use 3-dimensional variables for input/output transformations
    **kwargs (dict): Parameters for the attention function

  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.

    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hidden_dim] rather than the full memory.

  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionally returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.

  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  # import pdb; pdb.set_trace()


  if v_padding.lower() == "left":
    v_padding = "causal"

  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  vars_3d_num_heads = num_heads if vars_3d else 0



  if layer_collection is not None:
    if cache is not None:
      raise ValueError("KFAC implementation only supports cache is None.")
    if vars_3d:
      raise ValueError("KFAC implementation does not support 3d vars.")

  if recurrent_memory is not None:
    if memory_antecedent is not None:
      raise ValueError("Recurrent memory requires memory_antecedent is None.")
    if cache is not None:
      raise ValueError("Cache is not supported when using recurrent memory.")
    if vars_3d:
      raise ValueError("3d vars are not supported when using recurrent memory.")
    if layer_collection is not None:
      raise ValueError("KFAC is not supported when using recurrent memory.")
    if chunk_number is None:
      raise ValueError("chunk_number is required when using recurrent memory.")





  with tf.variable_scope(name, default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):

    if recurrent_memory is not None:
      (
          recurrent_memory_transaction,
          query_antecedent, memory_antecedent, bias,
      ) = recurrent_memory.pre_attention(
          chunk_number,
          query_antecedent, memory_antecedent, bias,
      )


    if cache is None or memory_antecedent is None:
      q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                            total_key_depth, total_value_depth, q_filter_width,
                            kv_filter_width, q_padding, kv_padding,
                            vars_3d_num_heads=vars_3d_num_heads,
                            v_kernel_size=v_kernel_size, v_dil_rate=v_dil_rate,
                            v_padding=v_padding, depthwise_sep=depthwise_sep,
                            layer_collection=layer_collection)

    if cache is not None:
      if attention_type not in ["dot_product", "dot_product_relative"]:
        # TODO(petershaw): Support caching when using relative position
        # representations, i.e. "dot_product_relative" attention.
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")

      if memory_antecedent is not None:
        # Encoder-Decoder Attention Cache
        q = compute_attention_component(query_antecedent, total_key_depth,
                                        q_filter_width, q_padding, "q",
                                        vars_3d_num_heads=vars_3d_num_heads)
        k = cache["k_encdec"]
        v = cache["v_encdec"]
      else:
        k = common_attention.split_heads(k, num_heads)
        v = common_attention.split_heads(v, num_heads)
        decode_loop_step = kwargs.get("decode_loop_step")
        if decode_loop_step is None:
          k = cache["k"] = tf.concat([cache["k"], k], axis=2)
          v = cache["v"] = tf.concat([cache["v"], v], axis=2)
        else:
          # Inplace update is required for inference on TPU.
          # Inplace_ops only supports inplace_update on the first dimension.
          # The performance of current implementation is better than updating
          # the tensor by adding the result of matmul(one_hot,
          # update_in_current_step)
          tmp_k = tf.transpose(cache["k"], perm=[2, 0, 1, 3])
          tmp_k = inplace_ops.alias_inplace_update(
              tmp_k, decode_loop_step, tf.squeeze(k, axis=2))
          k = cache["k"] = tf.transpose(tmp_k, perm=[1, 2, 0, 3])
          tmp_v = tf.transpose(cache["v"], perm=[2, 0, 1, 3])
          tmp_v = inplace_ops.alias_inplace_update(
              tmp_v, decode_loop_step, tf.squeeze(v, axis=2))
          v = cache["v"] = tf.transpose(tmp_v, perm=[1, 2, 0, 3])

    q = common_attention.split_heads(q, num_heads)
    if cache is None:
      k = common_attention.split_heads(k, num_heads)
      v = common_attention.split_heads(v, num_heads)

    key_depth_per_head = total_key_depth // num_heads
    if not vars_3d:
      q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack



    #  FIXME:STEP#1
    elif attention_type == "dot_product":
        x = common_attention.dot_product_attention(q, k, v, bias, dropout_rate, image_shapes,
                                  save_weights_to=save_weights_to,
                                  make_image_summary=make_image_summary,
                                  dropout_broadcast_dims=dropout_broadcast_dims,
                                  activation_dtype=kwargs.get(
                                      "activation_dtype"),
                                  hard_attention_k=hard_attention_k)
#     elif attention_type == "dot_product_relative":
#       x = dot_product_attention_relative(
#           q,
#           k,
#           v,
#           bias,
#           max_relative_position,
#           dropout_rate,
#           image_shapes,
#           save_weights_to=save_weights_to,
#           make_image_summary=make_image_summary,
#           cache=cache is not None,
#           allow_memory=recurrent_memory is not None,
#           hard_attention_k=hard_attention_k)
#     elif attention_type == "dot_product_unmasked_relative_v2":
#       x = dot_product_unmasked_self_attention_relative_v2(
#           q,
#           k,
#           v,
#           bias,
#           max_relative_position,
#           dropout_rate,
#           image_shapes,
#           make_image_summary=make_image_summary,
#           dropout_broadcast_dims=dropout_broadcast_dims,
#           heads_share_relative_embedding=heads_share_relative_embedding,
#           add_relative_to_values=add_relative_to_values)
#     elif attention_type == "dot_product_relative_v2":
#       x = dot_product_self_attention_relative_v2(
#           q,
#           k,
#           v,
#           bias,
#           max_relative_position,
#           dropout_rate,
#           image_shapes,
#           make_image_summary=make_image_summary,
#           dropout_broadcast_dims=dropout_broadcast_dims,
#           heads_share_relative_embedding=heads_share_relative_embedding,
#           add_relative_to_values=add_relative_to_values)



    #  FIXME:STEP#1  TODO:  max_length might need to be switched back to output_depth
    elif attention_type == "pointwise_attention":
      x = pointwise_attention(q, k, v, bias, dropout_rate, image_shapes,
                                save_weights_to=save_weights_to,
                                make_image_summary=make_image_summary,
                                dropout_broadcast_dims=dropout_broadcast_dims,
                                max_len=max_length, padding=v_padding, num_heads=num_heads)
                                #max_len=output_depth, padding=v_padding, num_heads=num_heads)


#     elif attention_type == "local_within_block_mask_right":
#       x = masked_within_block_local_attention_1d(
#           q, k, v, block_length=block_length)
#     elif attention_type == "local_relative_mask_right":
#       x = masked_relative_local_attention_1d(
#           q,
#           k,
#           v,
#           block_length=block_length,
#           make_image_summary=make_image_summary,
#           dropout_rate=dropout_rate,
#           heads_share_relative_embedding=heads_share_relative_embedding,
#           add_relative_to_values=add_relative_to_values,
#           name="masked_relative_local_attention_1d")
#     elif attention_type == "local_mask_right":
#       x = masked_local_attention_1d(
#           q,
#           k,
#           v,
#           block_length=block_length,
#           make_image_summary=make_image_summary)
#     elif attention_type == "local_unmasked":
#       x = local_attention_1d(
#           q, k, v, block_length=block_length, filter_width=block_width)
#     elif attention_type == "masked_dilated_1d":
#       x = masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
#                                            gap_size, num_memory_blocks)
    else:
      assert attention_type == "unmasked_dilated_1d"
      x = dilated_self_attention_1d(q, k, v, block_length, block_width,
                                    gap_size, num_memory_blocks)

    x = common_attention.combine_heads(x)

    # Set last dim specifically.
    x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

    if vars_3d:
      o_var = tf.get_variable(
          "o", [num_heads, total_value_depth // num_heads, output_depth])
      o_var = tf.cast(o_var, x.dtype)
      o_var = tf.reshape(o_var, [total_value_depth, output_depth])
      x = tf.tensordot(x, o_var, axes=1)
    else:
      x = common_layers.dense(
          x, output_depth, use_bias=False, name="output_transform")

      #  FIXME:STEP#1
      #  TODO:  test here, I've changed the output layer to a full conv1d layer






#################################################################################
###FIXME:  qurentined!!! debug ##############################################################################
#################################################################################
#################################################################################
#################################################################################
      # x = depthwise_sep_dilated_causal_conv1d(x, filters=output_depth,
      #                            kernel_size=v_kernel_size,
      #                            strides=1, dilation_rate=combine_dil_rate,
      #                            padding=v_padding,
      #                            depthwise_sep=depthwise_sep)

#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
      # x = dilated_causal_conv1d(x, filters=output_depth,
      #                            kernel_size=v_kernel_size,
      #                            strides=1, dilation_rate=combine_dil_rate,
      #                            padding=v_padding)

    if recurrent_memory is not None:
      x = recurrent_memory.post_attention(recurrent_memory_transaction, x)
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x

















































































































































def transformer_prepare_encoder(inputs, target_space, hparams, features=None):
  """Prepare one shard of the model for the encoder.

  Args:
    inputs: a Tensor.
    target_space: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well.
      This is needed now for "packed" datasets.

  Returns:
    encoder_input: a Tensor, bottom of encoder stack
    encoder_self_attention_bias: a bias tensor for use in encoder self-attention
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
  """

  print("\n\n\n")
  print("")
  print("TRANSFORMER PREPARE ENCODER!!")
  print("")
  print("\n\n\n")


  ishape_static = inputs.shape.as_list()
  encoder_input = inputs
  if features and "inputs_segmentation" in features:
    # Packed dataset.  Keep the examples from seeing each other.
    inputs_segmentation = features["inputs_segmentation"]
    inputs_position = features["inputs_position"]
    targets_segmentation = features["targets_segmentation"]
    if (hasattr(hparams, "unidirectional_encoder") and
        hparams.unidirectional_encoder):
      tf.logging.info("Using unidirectional encoder")
      encoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(inputs)[1]))
    else:
      encoder_self_attention_bias = (
          common_attention.attention_bias_same_segment(
              inputs_segmentation, inputs_segmentation))
    encoder_decoder_attention_bias = (
        common_attention.attention_bias_same_segment(targets_segmentation,
                                                     inputs_segmentation))
  else:
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    if (hasattr(hparams, "unidirectional_encoder") and
        hparams.unidirectional_encoder):
      tf.logging.info("Using unidirectional encoder")
      encoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(inputs)[1]))
    else:
      # Usual case - not a packed dataset.
      encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    inputs_position = None
  if hparams.proximity_bias:
    encoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(inputs)[1])
  if target_space is not None and hparams.get("use_target_space_embedding",
                                              True):
    # Append target_space_id embedding to inputs.
    emb_target_space = common_layers.embedding(
        target_space,
        32,
        ishape_static[-1],
        name="target_space_embedding",
        dtype=hparams.get("activation_dtype", "float32"))
    emb_target_space = tf.reshape(emb_target_space, [1, 1, -1])
    encoder_input += emb_target_space
  if hparams.pos == "timing":
    if inputs_position is not None:
      encoder_input = common_attention.add_timing_signal_1d_given_position(
          encoder_input, inputs_position)
    else:
      encoder_input = common_attention.add_timing_signal_1d(encoder_input)
  elif hparams.pos == "emb":
    encoder_input = common_attention.add_positional_embedding(
        encoder_input, hparams.max_length, "inputs_positional_embedding",
        inputs_position)

  encoder_self_attention_bias = common_layers.cast_like(
      encoder_self_attention_bias, encoder_input)
  encoder_decoder_attention_bias = common_layers.cast_like(
      encoder_decoder_attention_bias, encoder_input)
  return (encoder_input, encoder_self_attention_bias,
          encoder_decoder_attention_bias)

#################################################################
def transformer_encoder(encoder_input,
                        encoder_self_attention_bias,
                        hparams,
                        name="encoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        attn_bias_for_padding=None):
  """A stack of transformer layers.

  Args:
    encoder_input: a Tensor
    encoder_self_attention_bias: bias Tensor for self-attention
       (see common_attention.attention_bias())
    hparams: hyperparameters for model
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This must either be
      passed in, which we do for "packed" datasets, or inferred from
      encoder_self_attention_bias.  The knowledge about padding is used
      for pad_remover(efficiency) and to mask out padding in convolutional
      layers.
    save_weights_to: an optional dictionary to capture attention weights
      for visualization; the weights tensor will be appended there under
      a string key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    attn_bias_for_padding: Padded attention bias in case a unidirectional
      encoder is being used where future attention is masked.

  Returns:
    y: a Tensors
  """
  x = encoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_encoder_layers or hparams.num_hidden_layers)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      })

  with tf.variable_scope(name):
    if nonpadding is not None:
      padding = 1.0 - nonpadding
    else:
      attention_bias = encoder_self_attention_bias
      if attn_bias_for_padding is not None:
        attention_bias = attn_bias_for_padding
      padding = common_attention.attention_bias_to_padding(attention_bias)
      nonpadding = 1.0 - padding
    pad_remover = None
    if hparams.use_pad_remover and not common_layers.is_xla_compiled():
      pad_remover = expert_utils.PadRemover(padding)
    for layer in range(hparams.num_encoder_layers or hparams.num_hidden_layers):
      with tf.variable_scope("layer_%d" % layer):
        with tf.variable_scope("self_attention"):

          y = multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              encoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,

              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              vars_3d=hparams.get("attention_variables_3d"),

              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"),
              hard_attention_k=hparams.get("hard_attention_k", 0),

              v_kernel_size=hparams.conv_module_kernel_size,
              # v_padding="SAME"
              v_padding=hparams.conv_padding,
              depthwise_sep=hparams.depthwise_sep[layer])
          x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams),
              hparams,
              pad_remover,
              #conv_padding="SAME",
              conv_padding=hparams.conv_padding,
              nonpadding_mask=nonpadding,
              losses=losses,
              layer=layer)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(x, hparams)


def transformer_ffn_layer(x,
                          hparams,
                          pad_remover=None,
                          conv_padding="LEFT",
                          nonpadding_mask=None,
                          losses=None,
                          cache=None,
                          decode_loop_step=None,
                          readout_filter_size=0,
                          layer_collection=None,
                          layer=None):
  """Feed-forward layer in the transformer.

  Args:
    x: a Tensor of shape [batch_size, length, hparams.hidden_size]
    hparams: hyperparameters for model
    pad_remover: an expert_utils.PadRemover object tracking the padding
      positions. If provided, when using convolutional settings, the padding
      is removed before applying the convolution, and restored afterward. This
      can give a significant speedup.
    conv_padding: a string - either "LEFT" or "SAME".
    nonpadding_mask: an optional Tensor with shape [batch_size, length].
      needed for convolutional layers with "SAME" padding.
      Contains 1.0 in positions corresponding to nonpadding.
    losses: optional list onto which to append extra training losses
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop.
        Only used for inference on TPU.
    readout_filter_size: if it's greater than 0, then it will be used instead of
      filter_size
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.


  Returns:
    a Tensor of shape [batch_size, length, hparams.hidden_size]

  Raises:
    ValueError: If losses arg is None, but layer generates extra losses.
  """
  ffn_layer = hparams.ffn_layer
  relu_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "relu_dropout_broadcast_dims", "")))
  if ffn_layer == "conv_hidden_relu":
    # Backwards compatibility
    ffn_layer = "dense_relu_dense"
  if ffn_layer == "dense_relu_dense":
    # In simple convolution mode, use `pad_remover` to speed up processing.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_FILTER_DENSE,
        value={
            "filter_size": hparams.filter_size,
            "use_bias": "True",
            "activation": mlperf_log.RELU
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_FFN_OUTPUT_DENSE,
        value={
            "hidden_size": hparams.hidden_size,
            "use_bias": "True",
        })
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_RELU_DROPOUT, value=hparams.relu_dropout)
    if pad_remover:
      original_shape = common_layers.shape_list(x)
      # Collapse `x` across examples, and remove padding positions.
      x = tf.reshape(x, tf.concat([[-1], original_shape[2:]], axis=0))
      x = tf.expand_dims(pad_remover.remove(x), axis=0)
    conv_output = common_layers.dense_relu_dense(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        dropout=hparams.relu_dropout,
        dropout_broadcast_dims=relu_dropout_broadcast_dims,
        layer_collection=layer_collection)
    if pad_remover:
      # Restore `conv_output` to the original shape of `x`, including padding.
      conv_output = tf.reshape(
          pad_remover.restore(tf.squeeze(conv_output, axis=0)), original_shape)
    return conv_output
  elif ffn_layer == "conv_relu_conv":
    return conv_relu_conv(
        x,
        hparams.filter_size,
        hparams.hidden_size,
        first_kernel_size=hparams.conv_module_kernel_size,
        second_kernel_size=hparams.conv_module_kernel_size,
        padding=conv_padding,
        nonpadding_mask=nonpadding_mask,
        dropout=hparams.relu_dropout,
        cache=cache,
        decode_loop_step=decode_loop_step,
        #  TODO:  add back these dilations later
        dil_rate1=hparams.conv_module_dilations[layer][0],
        dil_rate2=hparams.conv_module_dilations[layer][1],
        depthwise_sep=hparams.depthwise_sep[layer])
  elif ffn_layer == "parameter_attention":
    return common_attention.parameter_attention(
        x, hparams.parameter_attention_key_channels or hparams.hidden_size,
        hparams.parameter_attention_value_channels or hparams.hidden_size,
        hparams.hidden_size, readout_filter_size or hparams.filter_size,
        hparams.num_heads,
        hparams.attention_dropout)
  elif ffn_layer == "conv_hidden_relu_with_sepconv":
    return common_layers.conv_hidden_relu(
        x,
        readout_filter_size or hparams.filter_size,
        hparams.hidden_size,
        kernel_size=(3, 1),
        second_kernel_size=(31, 1),
        padding="LEFT",
        dropout=hparams.relu_dropout)
  elif ffn_layer == "sru":
    return common_layers.sru(x)
  elif ffn_layer == "local_moe_tpu":
    overhead = hparams.moe_overhead_eval
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      overhead = hparams.moe_overhead_train
    ret, loss = expert_utils.local_moe_tpu(
        x,
        hparams.filter_size // 2,
        hparams.hidden_size,
        hparams.moe_num_experts,
        overhead=overhead,
        loss_coef=hparams.moe_loss_coef)
  elif ffn_layer == "local_moe":
    overhead = hparams.moe_overhead_eval
    if hparams.mode == tf.estimator.ModeKeys.TRAIN:
      overhead = hparams.moe_overhead_train
    ret, loss = expert_utils.local_moe(
        x,
        True,
        expert_utils.ffn_expert_fn(hparams.hidden_size, [hparams.filter_size],
                                   hparams.hidden_size),
        hparams.moe_num_experts,
        k=hparams.moe_k,
        hparams=hparams)
    losses.append(loss)
    return ret
  else:
    assert ffn_layer == "none"
    return x




































def transformer_encode(encoder_function, inputs, target_space, hparams,
                       attention_weights=None, features=None, losses=None,
                       **kwargs):
  """Encode transformer inputs.

  Args:
    encoder_function: the encoder function
    inputs: Transformer inputs [batch_size, input_length, 1, hidden_dim] which
      will be flattened along the two spatial dimensions.
    target_space: scalar, target space ID.
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.
    losses: optional list onto which to append extra training losses
    **kwargs: additional arguments to pass to encoder_function

  Returns:
    Tuple of:
        encoder_output: Encoder representation.
            [batch_size, input_length, hidden_dim]
        encoder_decoder_attention_bias: Bias and mask weights for
            encoder-decoder attention. [batch_size, input_length]
  """
  inputs = common_layers.flatten4d3d(inputs)

  encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
      transformer_prepare_encoder(
          inputs, target_space, hparams, features=features))

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)

  encoder_input = tf.nn.dropout(encoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  attn_bias_for_padding = None
  # Otherwise the encoder will just use encoder_self_attention_bias.
  if hparams.unidirectional_encoder:
    attn_bias_for_padding = encoder_decoder_attention_bias

  encoder_output = encoder_function(
      encoder_input,
      self_attention_bias,
      hparams,
      nonpadding=features_to_nonpadding(features, "inputs"),
      save_weights_to=attention_weights,
      make_image_summary=not common_layers.is_xla_compiled(),
      losses=losses,
      attn_bias_for_padding=attn_bias_for_padding,
      **kwargs)

  return encoder_output, encoder_decoder_attention_bias


def transformer_decode(decoder_function,
                       decoder_input,
                       encoder_output,
                       encoder_decoder_attention_bias,
                       decoder_self_attention_bias,
                       hparams,
                       attention_weights=None,
                       cache=None,
                       decode_loop_step=None,
                       nonpadding=None,
                       losses=None,
                       **kwargs):
  """Decode Transformer outputs from encoder representation.

  Args:
    decoder_function: the decoder function
    decoder_input: inputs to bottom of the model. [batch_size, decoder_length,
      hidden_dim]
    encoder_output: Encoder representation. [batch_size, input_length,
      hidden_dim]
    encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
      attention. [batch_size, input_length]
    decoder_self_attention_bias: Bias and mask weights for decoder
      self-attention. [batch_size, decoder_length]
    hparams: hyperparameters for model.
    attention_weights: weight to store attention to.
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    nonpadding: optional Tensor with shape [batch_size, decoder_length]
    losses: optional list onto which to append extra training losses
    **kwargs: additional arguments to pass to decoder_function

  Returns:
    Final decoder representation. [batch_size, decoder_length, hidden_dim]
  """
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_LAYER_POSTPROCESS_DROPOUT,
      value=hparams.layer_prepostprocess_dropout,
      hparams=hparams)  #  TODO:  maybe I want to comment this line out?
  decoder_input = tf.nn.dropout(decoder_input,
                                1.0 - hparams.layer_prepostprocess_dropout)

  decoder_output = decoder_function(
      decoder_input,
      encoder_output,
      decoder_self_attention_bias,
      encoder_decoder_attention_bias,
      hparams,
      cache=cache,
      decode_loop_step=decode_loop_step,
      nonpadding=nonpadding,
      save_weights_to=attention_weights,
      losses=losses,
      **kwargs)

  if (common_layers.is_xla_compiled() and
      hparams.mode == tf.estimator.ModeKeys.TRAIN):
    # TPU does not react kindly to extra dimensions.
    # TODO(noam): remove this once TPU is more forgiving of extra dims.
    return decoder_output
  else:
    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)



















#############################################################################################
@registry.register_model
class ConvTransformerApril2019(t2t_model.T2TModel):
  """Attention net.  See file docstring."""

  def __init__(self, *args, **kwargs):



    print("\n\n\n")
    print("")
    print("INIT CONVT!!")
    print("")
    print("\n\n\n")




    super(ConvTransformerApril2019, self).__init__(*args, **kwargs)
    self.attention_weights = {}  # For visualizing attention heads.
    self.recurrent_memory_by_layer = None  # Override to enable recurrent memory
    self._encoder_function = transformer_encoder
    self._decoder_function = transformer_decoder


  def encode(self, inputs, target_space, hparams, features=None, losses=None):
    """Encode transformer inputs, see transformer_encode."""
    return transformer_encode(
        self._encoder_function, inputs, target_space, hparams,
        attention_weights=self.attention_weights,
        features=features, losses=losses)

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None,
             decode_loop_step=None,
             nonpadding=None,
             losses=None,
             **kwargs):
    """Decode Transformer outputs, see transformer_decode."""
    return transformer_decode(
        self._decoder_function, decoder_input, encoder_output,
        encoder_decoder_attention_bias, decoder_self_attention_bias,
        hparams, attention_weights=self.attention_weights, cache=cache,
        decode_loop_step=decode_loop_step, nonpadding=nonpadding, losses=losses,
        **kwargs)




  def value_formation_module(self, input):
    module_output_size = input.get_shape()[-1]


    # input = tf.layers.conv1d(input, self._hparams.value_formation_module_hidden_dim,
    #               kernel_size=3, strides=1, padding="SAME", dilation_rate=1)
    input = tf.keras.layers.Conv1D(self._hparams.value_formation_module_hidden_dim,
                  kernel_size=3, strides=1, padding="SAME", dilation_rate=1)(input)
    # input = tf.nn.relu(batch_normalization(input, training=training))
    input = tf.nn.relu(input)

    # input = tf.layers.conv1d(input, module_output_size,
    #               kernel_size=1, strides=1, padding="SAME", dilation_rate=1)
    input = tf.keras.layers.Conv1D(module_output_size,
                  kernel_size=1, strides=1, padding="SAME", dilation_rate=1)(input)
    # return tf.nn.relu(batch_normalization(input, training=training))
    return tf.nn.relu(input)


  ####################################################
  def body(self, features):
    """Transformer main model_fn.

    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs. [batch_size, input_length, 1,
            hidden_dim].
          "targets": Target decoder outputs. [batch_size, decoder_length, 1,
            hidden_dim]
          "target_space_id": A scalar int from data_generators.problem.SpaceID.

    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    losses = []

    #import pdb; pdb.set_trace()

    if self.has_input:
      inputs = features["inputs"]
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams, features=features, losses=losses)
    else:
      encoder_output, encoder_decoder_attention_bias = (None, None)

    if hparams.auto_regression:
      targets = features["targets"]
      targets_shape = common_layers.shape_list(targets)
      targets = common_layers.flatten4d3d(targets)  #  using targets here
      decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
          targets, hparams, features=features)
    else:
      targets_shape = common_layers.shape_list(features["targets"])
      module_input = common_layers.flatten4d3d(inputs)
      module_input = self.value_formation_module(module_input)  #  using transformed input here
      decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
          module_input, hparams, features=features)


    # Not all subclasses of Transformer support keyword arguments related to
    # recurrent memory, so only pass these arguments if memory is enabled.
    decode_kwargs = {}
    if self.recurrent_memory_by_layer is not None:
      # TODO(kitaev): The chunk_number feature currently has the same shape as
      # "targets", but this is only for the purposes of sharing sharding code.
      # In fact every token within the batch must have the same chunk number.
      chunk_number_each_token = tf.squeeze(features["chunk_number"], (-1, -2))
      chunk_number_each_batch = chunk_number_each_token[:, 0]
      # Uncomment the code below to verify that tokens within a batch share the
      # same chunk number:
      # with tf.control_dependencies([
      #     tf.assert_equal(chunk_number_each_token,
      #                     chunk_number_each_batch[:, None])
      # ]):
      #   chunk_number_each_batch = tf.identity(chunk_number_each_batch)
      decode_kwargs = dict(
          recurrent_memory_by_layer=self.recurrent_memory_by_layer,
          chunk_number=chunk_number_each_batch,
          )

    decoder_output = self.decode(
        decoder_input,
        encoder_output,
        encoder_decoder_attention_bias,
        decoder_self_attention_bias,
        hparams,
        #  nonpadding=features_to_nonpadding(features, "targets"),
        nonpadding=None,
        losses=losses,
        **decode_kwargs
        )

    expected_attentions = features.get("expected_attentions")
    if expected_attentions is not None:
      attention_loss = common_attention.encoder_decoder_attention_loss(
          expected_attentions, self.attention_weights,
          hparams.expected_attention_loss_type,
          hparams.expected_attention_loss_multiplier)
      return decoder_output, {"attention_loss": attention_loss}

    #  TODO:
    #  TODO:
    #  FIXME:  I'm just letting it output whatever shape it outputs naturally
    #  FIXME:  the top of the modality will take care of the final layer to
    #  FIXME:  match the output shape needed
    # ret = tf.reshape(decoder_output, targets_shape)
    ret = decoder_output






    # FIXME:  FIXME:  I'm trying to print the number of parameters!!!
    print("\nNUMBER OF PARAMTERS: ")
    print(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    print("\n")
    # ret = tf.print(ret, ["\nNUMBER OF PARAMTERS: "])
    # ret = tf.print(ret, [np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])])
    # ret = tf.print(ret, ["\n"])


    if losses:
      return ret, {"extra_loss": tf.add_n(losses)}
    else:
      return ret
  ######################################################################
  def _greedy_infer(self, features, decode_length, use_tpu=False):
    """Fast version of greedy decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      use_tpu: A bool. Whether to build the inference graph for TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    # For real-valued modalities use the slow decode path for now.
    if (self._target_modality_is_real or
        self._hparams.self_attention_type != "dot_product"):
      return super(ConvTransformerApril2019, self)._greedy_infer(features, decode_length)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length)
      return self._fast_decode(features, decode_length)
  ####################################################
  def _beam_decode(self,
                   features,
                   decode_length,
                   beam_size,
                   top_beams,
                   alpha,
                   use_tpu=False):
    """Beam search decoding.

    Args:
      features: an map of string to `Tensor`
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.
      use_tpu: A bool, whether to do beam decode on TPU.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }
    """
    if (self._hparams.self_attention_type not in [
        "dot_product", "dot_product_relative"
    ]):
      # Caching is not guaranteed to work with attention types other than
      # dot_product.
      # TODO(petershaw): Support fast decoding when using relative
      # position representations, i.e. "dot_product_relative" attention.
      return self._beam_decode_slow(features, decode_length, beam_size,
                                    top_beams, alpha, use_tpu)
    with tf.variable_scope(self.name):
      if use_tpu:
        return self._fast_decode_tpu(features, decode_length, beam_size,
                                     top_beams, alpha)
      return self._fast_decode(features, decode_length, beam_size, top_beams,
                               alpha)
  ######################################################################
  def _fast_decode_tpu(self,
                       features,
                       decode_length,
                       beam_size=1,
                       top_beams=1,
                       alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding on TPU, uses beam search
    iff beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: A map of string to model features.
      decode_length: An integer, how many additional timesteps to decode.
      beam_size: An integer, number of beams.
      top_beams: An integer, how many of the beams to return.
      alpha: A float that controls the length penalty. Larger the alpha,
        stronger the preference for longer translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }.

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]
    target_vocab_size = self._problem_hparams.vocab_size["targets"]
    if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor

    if self.has_input:
      inputs = features["inputs"]
      if target_modality == modalities.ModalityType.CLASS_LABEL:
        decode_length = 1
      else:
        decode_length = (
            common_layers.shape_list(inputs)[1] + features.get(
                "decode_length", decode_length))

      # TODO(llion): Clean up this reshaping logic.
      inputs = tf.expand_dims(inputs, axis=1)
      if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
      s = common_layers.shape_list(inputs)
      batch_size = s[0]
      inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
      # _shard_features called to ensure that the variable names match
      inputs = self._shard_features({"inputs": inputs})["inputs"]
      input_modality = self._problem_hparams.modality["inputs"]
      input_vocab_size = self._problem_hparams.vocab_size["inputs"]
      if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
        input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
      modality_name = hparams.name.get(
          "inputs",
          modalities.get_name(input_modality))(hparams, input_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get("inputs",
                                    modalities.get_bottom(input_modality))
        inputs = dp(bottom, inputs, hparams, input_vocab_size)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = None
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None

      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]

    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "emb":
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length + 1, hparams.hidden_size]),
          hparams.max_length, "body/targets_positional_embedding", None)
    else:
      positional_encoding = None
  #############################################################
    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: A tensor, inputs ids to the decoder. [batch_size, 1].
        i: An integer, Step number of the decoding loop.

      Returns:
        A tensor, processed targets [batch_size, 1, hidden_dim].
      """


      print("\n\n\n")
      print("")
      print("PREPOCESS TARGETS!!!")
      print("")
      print("\n\n\n")




      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(target_modality))
        targets = dp(bottom, targets, hparams, target_vocab_size)[0]
      targets = common_layers.flatten4d3d(targets)

      # GO embeddings are all zero, this is because transformer_prepare_decoder
      # Shifts the targets along by one for the input which pads with zeros.
      # If the modality already maps GO to the zero embeddings this is not
      # needed.
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if positional_encoding is not None:
        positional_encoding_shape = positional_encoding.shape.as_list()
        targets += tf.slice(
            positional_encoding, [0, i, 0],
            [positional_encoding_shape[0], 1, positional_encoding_shape[2]])
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)
  ###########################################################
    def symbols_to_logits_tpu_fn(ids, i, cache):
      """Go from ids to logits for next symbol on TPU.

      Args:
        ids: A tensor, symbol IDs.
        i: An integer, step number of the decoding loop. Only used for inference
          on TPU.
        cache: A dict, containing tensors which are the results of previous
          attentions, used for fast decoding.

      Returns:
        ret: A tensor, computed logits.
        cache: A dict, containing tensors which are the results of previous
            attentions, used for fast decoding.
      """
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias_shape = decoder_self_attention_bias.shape.as_list()
      bias = tf.slice(decoder_self_attention_bias, [0, 0, i, 0],
                      [bias_shape[0], bias_shape[1], 1, bias_shape[3]])

      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode,
            targets,
            cache.get("encoder_output"),
            cache.get("encoder_decoder_attention_bias"),
            bias,
            hparams,
            cache,
            i,
            nonpadding=features_to_nonpadding(features, "targets"))
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        top = hparams.top.get("targets",
                              modalities.get_top(target_modality))
        logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          return tf.one_hot(
              tf.tile(
                  tf.slice(partial_targets, [0, i],
                           [partial_targets.shape.as_list()[0], 1]),
                  [beam_size]), vocab_size, 0.0, -1e9)

        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    ret = fast_decode_tpu(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_tpu_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_vocab_size,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length)
    if partial_targets is not None:
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret
  #############################################################################
  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0):
    """Fast decoding.

    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.

    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for longer translations.

    Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if beam_size == 1 or
              [batch_size, top_beams, <= decode_length]
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams
    target_modality = self._problem_hparams.modality["targets"]
    target_vocab_size = self._problem_hparams.vocab_size["targets"]
    if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
      target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
    if "targets_segmentation" in features:
      raise NotImplementedError(
          "Decoding not supported on packed datasets "
          " If you want to decode from a dataset, use the non-packed version"
          " of the dataset when decoding.")
    if self.has_input:
      inputs = features["inputs"]
      if target_modality == modalities.ModalityType.CLASS_LABEL:
        decode_length = 1
      else:
        decode_length = (
            common_layers.shape_list(inputs)[1] + features.get(
                "decode_length", decode_length))

      # TODO(llion): Clean up this reshaping logic.
      inputs = tf.expand_dims(inputs, axis=1)
      if len(inputs.shape) < 5:
        inputs = tf.expand_dims(inputs, axis=4)
      s = common_layers.shape_list(inputs)
      batch_size = s[0]
      inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
      # _shard_features called to ensure that the variable names match
      inputs = self._shard_features({"inputs": inputs})["inputs"]
      input_modality = self._problem_hparams.modality["inputs"]
      input_vocab_size = self._problem_hparams.vocab_size["inputs"]
      if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
        input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
      modality_name = hparams.name.get(
          "inputs",
          modalities.get_name(input_modality))(hparams, input_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get("inputs",
                                    modalities.get_bottom(input_modality))
        inputs = dp(bottom, inputs, hparams, input_vocab_size)
      with tf.variable_scope("body"):
        encoder_output, encoder_decoder_attention_bias = dp(
            self.encode,
            inputs,
            features["target_space_id"],
            hparams,
            features=features)
      encoder_output = encoder_output[0]
      encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
      partial_targets = None
    else:
      # The problem has no inputs.
      encoder_output = None
      encoder_decoder_attention_bias = None

      # Prepare partial targets.
      # In either features["inputs"] or features["targets"].
      # We force the outputs to begin with these sequences.
      partial_targets = features.get("inputs")
      if partial_targets is None:
        partial_targets = features["targets"]
      assert partial_targets is not None
      partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
      partial_targets = tf.to_int64(partial_targets)
      partial_targets_shape = common_layers.shape_list(partial_targets)
      partial_targets_length = partial_targets_shape[1]
      decode_length = (
          partial_targets_length + features.get("decode_length", decode_length))
      batch_size = partial_targets_shape[0]

    if hparams.pos == "timing":
      positional_encoding = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)
    elif hparams.pos == "emb":
      positional_encoding = common_attention.add_positional_embedding(
          tf.zeros([1, decode_length, hparams.hidden_size]), hparams.max_length,
          "body/targets_positional_embedding", None)
    else:
      positional_encoding = None
  ##############################################################
    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.

      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.

      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.

      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """



      print("\n\n\n")
      print("")
      print("PREPROCESS TARGETS.....AGAIN?!!")
      print("")
      print("\n\n\n")






      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        bottom = hparams.bottom.get(
            "targets", modalities.get_targets_bottom(target_modality))
        targets = dp(bottom, targets, hparams, target_vocab_size)[0]
      targets = common_layers.flatten4d3d(targets)

      # GO embeddings are all zero, this is because transformer_prepare_decoder
      # Shifts the targets along by one for the input which pads with zeros.
      # If the modality already maps GO to the zero embeddings this is not
      # needed.
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if positional_encoding is not None:
        targets += positional_encoding[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)
  ###############################################################################
    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(
            self.decode,
            targets,
            cache.get("encoder_output"),
            cache.get("encoder_decoder_attention_bias"),
            bias,
            hparams,
            cache,
            nonpadding=features_to_nonpadding(features, "targets"))

      modality_name = hparams.name.get(
          "targets",
          modalities.get_name(target_modality))(hparams, target_vocab_size)
      with tf.variable_scope(modality_name):
        top = hparams.top.get("targets", modalities.get_top(target_modality))
        logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]

      ret = tf.squeeze(logits, axis=[1, 2, 3])
      if partial_targets is not None:
        # If the position is within the given partial targets, we alter the
        # logits to always return those values.
        # A faster approach would be to process the partial targets in one
        # iteration in order to fill the corresponding parts of the cache.
        # This would require broader changes, though.
        vocab_size = tf.shape(ret)[1]

        def forced_logits():
          return tf.one_hot(
              tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
              -1e9)

        ret = tf.cond(
            tf.less(i, partial_targets_length), forced_logits, lambda: ret)
      return ret, cache

    ret = fast_decode(
        encoder_output=encoder_output,
        encoder_decoder_attention_bias=encoder_decoder_attention_bias,
        symbols_to_logits_fn=symbols_to_logits_fn,
        hparams=hparams,
        decode_length=decode_length,
        vocab_size=target_vocab_size,
        beam_size=beam_size,
        top_beams=top_beams,
        alpha=alpha,
        batch_size=batch_size,
        force_decode_length=self._decode_hparams.force_decode_length)
    if partial_targets is not None:
      if beam_size <= 1 or top_beams <= 1:
        ret["outputs"] = ret["outputs"][:, partial_targets_length:]
      else:
        ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
    return ret




















































































































































































# def fast_decode_tpu(encoder_output,
#                     encoder_decoder_attention_bias,
#                     symbols_to_logits_fn,
#                     hparams,
#                     decode_length,
#                     vocab_size,
#                     beam_size=1,
#                     top_beams=1,
#                     alpha=1.0,
#                     sos_id=0,
#                     eos_id=beam_search.EOS_ID,
#                     batch_size=None,
#                     force_decode_length=False,
#                     scope_prefix="body/",
#                     use_top_k_with_unique=True):
#   """Given encoder output and a symbols to logits function, does fast decoding.
#
#   Implements both greedy and beam search decoding for TPU, uses beam search iff
#   beam_size > 1, otherwise beam search related arguments are ignored.
#
#   Args:
#     encoder_output: A tensor, output from encoder.
#     encoder_decoder_attention_bias: A tensor, bias for use in encoder-decoder
#       attention.
#     symbols_to_logits_fn: Incremental decoding, function mapping triple `(ids,
#       step, cache)` to symbol logits.
#     hparams: Run hyperparameters.
#     decode_length: An integer, how many additional timesteps to decode.
#     vocab_size: Output vocabulary size.
#     beam_size: An integer, number of beams.
#     top_beams: An integer, how many of the beams to return.
#     alpha: A float that controls the length penalty. Larger the alpha, stronger
#       the preference for longer translations.
#     sos_id: Start-of-sequence symbol.
#     eos_id: End-of-sequence symbol.
#     batch_size: An integer, must be passed if there is no input.
#     force_decode_length: A bool, whether to force the full decode length, or if
#       False, stop when all beams hit eos_id.
#     scope_prefix: str, prefix for decoder layer variable scopes.
#     use_top_k_with_unique: bool, whether to use a fast (but decreased precision)
#       top_k during beam search.
#
#   Returns:
#     A dict of decoding results {
#         "outputs": integer `Tensor` of decoded ids of shape
#             [batch_size, <= decode_length] if top_beams == 1 or
#             [batch_size, top_beams, <= decode_length] otherwise
#         "scores": decoding log probs from the beam search,
#             None if using greedy decoding (beam_size=1)
#     }.
#
#   Raises:
#     NotImplementedError: If beam size > 1 with partial targets.
#   """
#   if encoder_output is not None:
#     batch_size = common_layers.shape_list(encoder_output)[0]
#
#   key_channels = hparams.attention_key_channels or hparams.hidden_size
#   value_channels = hparams.attention_value_channels or hparams.hidden_size
#   num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
#   vars_3d_num_heads = (
#       hparams.num_heads if hparams.get("attention_variables_3d") else 0)
#
#   cache = {
#       "layer_%d" % layer: {  # pylint: disable=g-complex-comprehension
#           "k":
#           common_attention.split_heads(
#               tf.zeros([batch_size, decode_length, key_channels]),
#               hparams.num_heads),
#           "v":
#           common_attention.split_heads(
#               tf.zeros([batch_size, decode_length, value_channels]),
#               hparams.num_heads),
#       } for layer in range(num_layers)
#   }
#
#   # If `ffn_layer` is in `["dense_relu_dense" or "conv_hidden_relu"]`, then the
#   # cache key "f" won't be used, which means that the` shape of cache["f"]`
#   # won't be changed to
#   # `[beamsize*batch_size, decode_length, hparams.hidden_size]` and may cause
#   # error when applying `nest.map reshape function` on it.
#   if hparams.ffn_layer not in ["dense_relu_dense", "conv_hidden_relu"]:
#     for layer in range(num_layers):
#       cache["layer_%d" % layer]["f"] = tf.zeros(
#           [batch_size, 0, hparams.hidden_size])
#
#   if encoder_output is not None:
#     for layer in range(num_layers):
#       layer_name = "layer_%d" % layer
#       with tf.variable_scope("%sdecoder/%s/encdec_attention/multihead_attention"
#                              % (scope_prefix, layer_name)):
#         k_encdec = common_attention.compute_attention_component(
#             encoder_output,
#             key_channels,
#             name="k",
#             vars_3d_num_heads=vars_3d_num_heads)
#         k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads)
#         v_encdec = common_attention.compute_attention_component(
#             encoder_output,
#             value_channels,
#             name="v",
#             vars_3d_num_heads=vars_3d_num_heads)
#         v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads)
#       cache[layer_name]["k_encdec"] = k_encdec
#       cache[layer_name]["v_encdec"] = v_encdec
#
#     cache["encoder_output"] = encoder_output
#     cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias
#
#   mlperf_log.transformer_print(
#       key=mlperf_log.MODEL_HP_SEQ_BEAM_SEARCH,
#       value={
#           "vocab_size": vocab_size,
#           "batch_size": batch_size,
#           "beam_size": beam_size,
#           "alpha": alpha,
#           "max_decode_length": decode_length
#       },
#       hparams=hparams)
#   if beam_size > 1:  # Beam Search
#     initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
#     decoded_ids, scores, _ = beam_search.beam_search(
#         symbols_to_logits_fn,
#         initial_ids,
#         beam_size,
#         decode_length,
#         vocab_size,
#         alpha,
#         states=cache,
#         eos_id=eos_id,
#         stop_early=(top_beams == 1),
#         use_tpu=True,
#         use_top_k_with_unique=use_top_k_with_unique)
#
#     if top_beams == 1:
#       decoded_ids = decoded_ids[:, 0, 1:]
#       scores = scores[:, 0]
#     else:
#       decoded_ids = decoded_ids[:, :top_beams, 1:]
#       scores = scores[:, :top_beams]
#   else:  # Greedy
#
#     def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
#       """One step of greedy decoding."""
#       logits, cache = symbols_to_logits_fn(next_id, i, cache)
#       log_probs = common_layers.log_prob_from_logits(logits)
#       temperature = getattr(hparams, "sampling_temp", 0.0)
#       keep_top = getattr(hparams, "sampling_keep_top_k", -1)
#       if hparams.sampling_method == "argmax":
#         temperature = 0.0
#       next_id = common_layers.sample_with_temperature(
#           logits, temperature, keep_top)
#
#       hit_eos |= tf.equal(next_id, eos_id)
#
#       log_prob_indices = tf.stack([tf.range(tf.to_int64(batch_size)), next_id],
#                                   axis=1)
#       log_prob += tf.gather_nd(log_probs, log_prob_indices)
#
#       next_id = tf.expand_dims(next_id, axis=1)
#       decoded_ids = tf.transpose(decoded_ids)
#       decoded_ids = inplace_ops.alias_inplace_update(
#           decoded_ids, i, tf.squeeze(next_id, axis=1))
#       decoded_ids = tf.transpose(decoded_ids)
#       return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob
#
#     def is_not_finished(i, hit_eos, *_):
#       finished = i >= decode_length
#       if not force_decode_length:
#         finished |= tf.reduce_all(hit_eos)
#       return tf.logical_not(finished)
#
#     decoded_ids = tf.zeros([batch_size, decode_length], dtype=tf.int64)
#     hit_eos = tf.fill([batch_size], False)
#     next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
#     initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
#
#     def compute_cache_shape_invariants(tensor):
#       return tf.TensorShape(tensor.shape.as_list())
#
#     _, _, _, decoded_ids, _, log_prob = tf.while_loop(
#         is_not_finished,
#         inner_loop, [
#             tf.constant(0), hit_eos, next_id, decoded_ids, cache,
#             initial_log_prob
#         ],
#         shape_invariants=[
#             tf.TensorShape([]),
#             tf.TensorShape([batch_size]),
#             tf.TensorShape([batch_size, 1]),
#             tf.TensorShape([batch_size, decode_length]),
#             nest.map_structure(compute_cache_shape_invariants, cache),
#             tf.TensorShape([batch_size]),
#         ])
#     scores = log_prob
#
#   return {"outputs": decoded_ids, "scores": scores}
#
#
def fast_decode(encoder_output,
                encoder_decoder_attention_bias,
                symbols_to_logits_fn,
                hparams,
                decode_length,
                vocab_size,
                beam_size=1,
                top_beams=1,
                alpha=1.0,
                sos_id=0,
                eos_id=beam_search.EOS_ID,
                batch_size=None,
                force_decode_length=False,
                scope_prefix="body/",
                cache=None):
  """Given encoder output and a symbols to logits function, does fast decoding.

  Implements both greedy and beam search decoding, uses beam search iff
  beam_size > 1, otherwise beam search related arguments are ignored.

  Args:
    encoder_output: Output from encoder.
    encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
      attention
    symbols_to_logits_fn: Incremental decoding; function mapping triple `(ids,
      step, cache)` to symbol logits.
    hparams: run hyperparameters
    decode_length: an integer.  How many additional timesteps to decode.
    vocab_size: Output vocabulary size.
    beam_size: number of beams.
    top_beams: an integer. How many of the beams to return.
    alpha: Float that controls the length penalty. larger the alpha, stronger
      the preference for longer translations.
    sos_id: End-of-sequence symbol in beam search.
    eos_id: End-of-sequence symbol in beam search.
    batch_size: an integer scalar - must be passed if there is no input
    force_decode_length: bool, whether to force the full decode length, or if
      False, stop when all beams hit eos_id.
    scope_prefix: str, prefix for decoder layer variable scopes.
    cache: cache dictionary for additional predictions.

  Returns:
      A dict of decoding results {
          "outputs": integer `Tensor` of decoded ids of shape
              [batch_size, <= decode_length] if top_beams == 1 or
              [batch_size, top_beams, <= decode_length] otherwise
          "scores": decoding log probs from the beam search,
              None if using greedy decoding (beam_size=1)
      }

    Raises:
      NotImplementedError: If beam size > 1 with partial targets.
  """
  if encoder_output is not None:
    batch_size = common_layers.shape_list(encoder_output)[0]

  key_channels = hparams.attention_key_channels or hparams.hidden_size
  value_channels = hparams.attention_value_channels or hparams.hidden_size
  num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers
  vars_3d_num_heads = (
      hparams.num_heads if hparams.get("attention_variables_3d") else 0)

  if cache is None:
    cache = {}
  cache.update({
      "layer_%d" % layer: {  # pylint: disable=g-complex-comprehension
          "k":
              common_attention.split_heads(
                  tf.zeros([batch_size, 0, key_channels]), hparams.num_heads),
          "v":
              common_attention.split_heads(
                  tf.zeros([batch_size, 0, value_channels]), hparams.num_heads),
      } for layer in range(num_layers)
  })

  # If `ffn_layer` is in `["dense_relu_dense" or "conv_hidden_relu"]`, then the
  # cache key "f" won't be used, which means that the` shape of cache["f"]`
  # won't be changed to
  # `[beamsize*batch_size, decode_length, hparams.hidden_size]` and may cause
  # error when applying `nest.map reshape function` on it.
  if hparams.ffn_layer not in ["dense_relu_dense", "conv_hidden_relu"]:
    for layer in range(num_layers):
      cache["layer_%d" % layer]["f"] = tf.zeros(
          [batch_size, 0, hparams.hidden_size])

  if encoder_output is not None:
    for layer in range(num_layers):
      layer_name = "layer_%d" % layer
      with tf.variable_scope("%sdecoder/%s/encdec_attention/multihead_attention"
                             % (scope_prefix, layer_name)):
        k_encdec = common_attention.compute_attention_component(
            encoder_output,
            key_channels,
            name="k",
            vars_3d_num_heads=vars_3d_num_heads)
        k_encdec = common_attention.split_heads(k_encdec, hparams.num_heads)
        v_encdec = common_attention.compute_attention_component(
            encoder_output,
            value_channels,
            name="v",
            vars_3d_num_heads=vars_3d_num_heads)
        v_encdec = common_attention.split_heads(v_encdec, hparams.num_heads)
      cache[layer_name]["k_encdec"] = k_encdec
      cache[layer_name]["v_encdec"] = v_encdec

    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

  if beam_size > 1:  # Beam Search
    initial_ids = sos_id * tf.ones([batch_size], dtype=tf.int32)
    decoded_ids, scores, cache = beam_search.beam_search(
        symbols_to_logits_fn,
        initial_ids,
        beam_size,
        decode_length,
        vocab_size,
        alpha,
        states=cache,
        eos_id=eos_id,
        stop_early=(top_beams == 1))

    if top_beams == 1:
      decoded_ids = decoded_ids[:, 0, 1:]
      scores = scores[:, 0]
    else:
      decoded_ids = decoded_ids[:, :top_beams, 1:]
      scores = scores[:, :top_beams]
  else:  # Greedy

    def inner_loop(i, hit_eos, next_id, decoded_ids, cache, log_prob):
      """One step of greedy decoding."""
      logits, cache = symbols_to_logits_fn(next_id, i, cache)
      log_probs = common_layers.log_prob_from_logits(logits)
      temperature = getattr(hparams, "sampling_temp", 0.0)
      keep_top = getattr(hparams, "sampling_keep_top_k", -1)
      if hparams.sampling_method == "argmax":
        temperature = 0.0
      next_id = common_layers.sample_with_temperature(
          logits, temperature, keep_top)
      hit_eos |= tf.equal(next_id, eos_id)

      log_prob_indices = tf.stack([tf.range(tf.to_int64(batch_size)), next_id],
                                  axis=1)
      log_prob += tf.gather_nd(log_probs, log_prob_indices)

      next_id = tf.expand_dims(next_id, axis=1)
      decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
      return i + 1, hit_eos, next_id, decoded_ids, cache, log_prob

    def is_not_finished(i, hit_eos, *_):
      finished = i >= decode_length
      if not force_decode_length:
        finished |= tf.reduce_all(hit_eos)
      return tf.logical_not(finished)

    decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
    hit_eos = tf.fill([batch_size], False)
    next_id = sos_id * tf.ones([batch_size, 1], dtype=tf.int64)
    initial_log_prob = tf.zeros([batch_size], dtype=tf.float32)
    _, _, _, decoded_ids, cache, log_prob = tf.while_loop(
        is_not_finished,
        inner_loop, [
            tf.constant(0), hit_eos, next_id, decoded_ids, cache,
            initial_log_prob
        ],
        shape_invariants=[
            tf.TensorShape([]),
            tf.TensorShape([None]),
            tf.TensorShape([None, None]),
            tf.TensorShape([None, None]),
            nest.map_structure(beam_search.get_state_shape_invariants, cache),
            tf.TensorShape([None]),
        ])
    scores = log_prob

  return {"outputs": decoded_ids, "scores": scores, "cache": cache}


# @registry.register_model
# class TransformerScorer(Transformer):
#   """Transformer model, but only scores in PREDICT mode.
#
#   Checkpoints between Transformer and TransformerScorer are interchangeable.
#   """
#
#   def __init__(self, *args, **kwargs):
#     super(TransformerScorer, self).__init__(*args, **kwargs)
#     self._name = "transformer"
#     self._base_name = "transformer"
#
#   def infer(self,
#             features=None,
#             decode_length=50,
#             beam_size=1,
#             top_beams=1,
#             alpha=0.0,
#             use_tpu=False):
#     """Returns the targets and their log probabilities."""
#     del decode_length, beam_size, top_beams, alpha, use_tpu
#     assert features is not None
#
#     # Run the model
#     self.hparams.force_full_predict = True
#     with tf.variable_scope(self.name):
#       logits, _ = self.model_fn(features)
#     assert len(logits.shape) == 5  # [batch, time, 1, 1, vocab]
#     logits = tf.squeeze(logits, [2, 3])
#
#     # Compute the log probabilities
#     log_probs = common_layers.log_prob_from_logits(logits)
#
#     targets = features["targets"]
#     assert len(targets.shape) == 4  # [batch, time, 1, 1]
#     targets = tf.squeeze(targets, [2, 3])
#
#     # Slice out the log_probs of the targets
#     log_probs = common_layers.index_last_dim_with_indices(log_probs, targets)
#
#     # Sum over time to get the log_prob of the sequence
#     scores = tf.reduce_sum(log_probs, axis=1)
#
#     return {"outputs": targets, "scores": scores}


# @registry.register_model
# class TransformerEncoder(t2t_model.T2TModel):
#   """Transformer, encoder only."""
#
#   def body(self, features):
#     hparams = self._hparams
#     inputs = features["inputs"]
#     target_space = features["target_space_id"]
#
#     inputs = common_layers.flatten4d3d(inputs)
#
#     (encoder_input, encoder_self_attention_bias, _) = (
#         transformer_prepare_encoder(inputs, target_space, hparams))
#
#     encoder_input = tf.nn.dropout(encoder_input,
#                                   1.0 - hparams.layer_prepostprocess_dropout)
#     encoder_output = transformer_encoder(
#         encoder_input,
#         encoder_self_attention_bias,
#         hparams,
#         nonpadding=features_to_nonpadding(features, "inputs"))
#     encoder_output = tf.expand_dims(encoder_output, 2)
#
#     return encoder_output


# @registry.register_model
# class TransformerRegressor(TransformerEncoder):
#   """Transformer inheriting from Encoder, for the regression problem.
#
#   Final result is a tensor that has a shape of (?, 1, 1, 1).
#   """
#
#   def top(self, body_output, features):
#     """Computes single scalar value from body_output."""
#
#     with tf.variable_scope("reg_top_ffn"):
#       x = body_output
#       x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
#       res = tf.layers.dense(x, 1, name="model_top")
#       return res


def features_to_nonpadding(features, inputs_or_targets="inputs"):
  key = inputs_or_targets + "_segmentation"
  if features and key in features:
    return tf.minimum(tf.to_float(features[key]), 1.0)
  return None


def transformer_prepare_decoder(targets, hparams, features=None):
  """Prepare one shard of the model for the decoder.

  Args:
    targets: a Tensor.
    hparams: run hyperparameters
    features: optionally pass the entire features dictionary as well. This is
      needed now for "packed" datasets.

  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_self_attention_bias: a bias tensor for use in decoder self-attention
  """


  print("\n\n\n")
  print("")
  print("TRANSFORMER PREPARE DECODER!!!")
  print("")
  print("\n\n\n")






  if hparams.causal_decoder_self_attention:
    # Causal attention.
    if hparams.prepend_mode == "prepend_inputs_full_attention":
      decoder_self_attention_bias = (
          common_attention.attention_bias_prepend_inputs_full_attention(
              common_attention.embedding_to_padding(targets)))
    else:
      decoder_self_attention_bias = (
          common_attention.attention_bias_lower_triangle(
              common_layers.shape_list(targets)[1]))
  else:
    # Full attention.
    decoder_padding = common_attention.embedding_to_padding(targets)
    decoder_self_attention_bias = (
        common_attention.attention_bias_ignore_padding(decoder_padding))

  if features and "targets_segmentation" in features:
    # "Packed" dataset - keep the examples from seeing each other.
    targets_segmentation = features["targets_segmentation"]
    targets_position = features["targets_position"]
    decoder_self_attention_bias += common_attention.attention_bias_same_segment(
        targets_segmentation, targets_segmentation)
  else:
    targets_position = None
  if hparams.proximity_bias:
    decoder_self_attention_bias += common_attention.attention_bias_proximal(
        common_layers.shape_list(targets)[1])
  decoder_input = common_layers.shift_right_3d(targets)
  if hparams.pos == "timing":
    if targets_position is not None:
      decoder_input = common_attention.add_timing_signal_1d_given_position(
          decoder_input, targets_position)
    else:
      decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  elif hparams.pos == "emb":
    decoder_input = common_attention.add_positional_embedding(
        decoder_input, hparams.max_length, "targets_positional_embedding",
        targets_position)

  if hparams.activation_dtype == "bfloat16":
    decoder_self_attention_bias = tf.cast(decoder_self_attention_bias,
                                          tf.bfloat16)
  return (decoder_input, decoder_self_attention_bias)


def transformer_decoder(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        decode_loop_step=None,
                        name="decoder",
                        nonpadding=None,
                        save_weights_to=None,
                        make_image_summary=True,
                        losses=None,
                        layer_collection=None,
                        recurrent_memory_by_layer=None,
                        chunk_number=None,
                        ):
  """A stack of transformer layers.

  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_self_attention_bias: bias Tensor for self-attention (see
      common_attention.attention_bias())
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
      attentions, used for fast decoding.
    decode_loop_step: An integer, step number of the decoding loop. Only used
      for inference on TPU.
    name: a string
    nonpadding: optional Tensor with shape [batch_size, encoder_length]
      indicating what positions are not padding.  This is used to mask out
      padding in convolutional layers.  We generally only need this mask for
      "packed" datasets, because for ordinary datasets, no padding is ever
      followed by nonpadding.
    save_weights_to: an optional dictionary to capture attention weights for
      visualization; the weights tensor will be appended there under a string
      key created from the variable scope (including name).
    make_image_summary: Whether to make an attention image summary.
    losses: optional list onto which to append extra training losses
    layer_collection: A tensorflow_kfac.LayerCollection. Only used by the
      KFAC optimizer. Default is None.
    recurrent_memory_by_layer: Optional dict, mapping layer names to instances
      of transformer_memory.RecurrentMemory. Default is None.
    chunk_number: an optional integer Tensor with shape [batch] used to operate
      the recurrent_memory.

  Returns:
    y: a Tensors
  """
  x = decoder_input
  attention_dropout_broadcast_dims = (
      common_layers.comma_separated_string_to_integer_list(
          getattr(hparams, "attention_dropout_broadcast_dims", "")))

  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_NUM_HIDDEN_LAYERS,
      value=hparams.num_decoder_layers or hparams.num_hidden_layers,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DROPOUT,
      value=hparams.attention_dropout,
      hparams=hparams)
  mlperf_log.transformer_print(
      key=mlperf_log.MODEL_HP_ATTENTION_DENSE,
      value={
          "use_bias": "false",
          "num_heads": hparams.num_heads,
          "hidden_size": hparams.hidden_size
      },
      hparams=hparams)

  with tf.variable_scope(name):
    for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      if recurrent_memory_by_layer is not None:
        recurrent_memory = recurrent_memory_by_layer[layer_name]
      else:
        recurrent_memory = None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y = multihead_attention(
              common_layers.layer_preprocess(
                  x, hparams, layer_collection=layer_collection),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,

              max_relative_position=hparams.max_relative_position,
              heads_share_relative_embedding=(
                  hparams.heads_share_relative_embedding),
              add_relative_to_values=hparams.add_relative_to_values,
              save_weights_to=save_weights_to,
              cache=layer_cache,
              make_image_summary=make_image_summary,
              dropout_broadcast_dims=attention_dropout_broadcast_dims,
              max_length=hparams.get("max_length"),
              decode_loop_step=decode_loop_step,
              vars_3d=hparams.get("attention_variables_3d"),

              activation_dtype=hparams.get("activation_dtype", "float32"),
              weight_dtype=hparams.get("weight_dtype", "float32"),
              layer_collection=layer_collection,
              recurrent_memory=recurrent_memory,
              chunk_number=chunk_number,
              hard_attention_k=hparams.get("hard_attention_k", 0),
              v_kernel_size=hparams.conv_module_kernel_size,
              # v_padding=hparams.conv_padding
              v_padding="CAUSAL",
              depthwise_sep=hparams.depthwise_sep[layer])
          x = common_layers.layer_postprocess(x, y, hparams)
        if encoder_output is not None:
          with tf.variable_scope("encdec_attention"):
            y = multihead_attention(
                common_layers.layer_preprocess(
                    x, hparams, layer_collection=layer_collection),
                encoder_output,
                encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size,
                hparams.num_heads,
                hparams.attention_dropout,
                max_relative_position=hparams.max_relative_position,
                heads_share_relative_embedding=(
                    hparams.heads_share_relative_embedding),
                add_relative_to_values=hparams.add_relative_to_values,
                save_weights_to=save_weights_to,
                cache=layer_cache,
                make_image_summary=make_image_summary,
                dropout_broadcast_dims=attention_dropout_broadcast_dims,
                max_length=hparams.get("max_length"),
                vars_3d=hparams.get("attention_variables_3d"),
                activation_dtype=hparams.get("activation_dtype", "float32"),
                weight_dtype=hparams.get("weight_dtype", "float32"),
                layer_collection=layer_collection,
                hard_attention_k=hparams.get("hard_attention_k", 0),
                v_kernel_size=hparams.conv_module_kernel_size,
                # v_padding=hparams.conv_padding
                v_padding="CAUSAL",
                depthwise_sep=hparams.depthwise_sep[layer])
            x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer_ffn_layer(
              common_layers.layer_preprocess(
                  x, hparams, layer_collection=layer_collection),
              hparams,
              conv_padding="CAUSAL",
              nonpadding_mask=nonpadding,
              losses=losses,
              cache=layer_cache,
              decode_loop_step=decode_loop_step,
              layer_collection=layer_collection,
              layer=layer)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it should also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    mlperf_log.transformer_print(
        key=mlperf_log.MODEL_HP_NORM,
        value={"hidden_size": hparams.hidden_size})
    return common_layers.layer_preprocess(
        x, hparams, layer_collection=layer_collection)

#
# @registry.register_model
# class TransformerMemory(Transformer):
#   """Transformer language model with memory across chunks."""
#
#   # TODO(kitaev): consider overriding set_mode to swap out recurrent memory when
#   # switching between training and evaluation.
#
#   def __init__(self, *args, **kwargs):
#     super(TransformerMemory, self).__init__(*args, **kwargs)
#
#     hparams = self._hparams
#     self.recurrent_memory_by_layer = {}
#     for layer in range(hparams.num_decoder_layers or hparams.num_hidden_layers):
#       layer_name = "layer_%d" % layer
#       if hparams.memory_type == "neural_memory":
#         memory = transformer_memory.TransformerMemory(
#             batch_size=int(hparams.batch_size / hparams.max_length),
#             key_depth=hparams.hidden_size,
#             val_depth=hparams.hidden_size,
#             memory_size=hparams.split_targets_chunk_length,
#             sharpen_factor=1.,
#             name=layer_name + "/recurrent_memory")
#       elif hparams.memory_type == "transformer_xl":
#         memory = transformer_memory.RecentTokensMemory(
#             layer_name + "/recurrent_memory", hparams)
#       else:
#         raise ValueError("Unsupported memory type: %s" % hparams.memory_type)
#       self.recurrent_memory_by_layer[layer_name] = memory
#
#   @property
#   def has_input(self):
#     if hasattr(self._hparams, "unconditional") and self._hparams.unconditional:
#       return False
#     return super(TransformerMemory, self).has_input
#
#   def _beam_decode(self, features, decode_length, beam_size, top_beams, alpha,
#                    use_tpu=False):
#     """Overriding beam search because for now only the slow version works with
#     memory
#     """
#     return self._beam_decode_slow(features, decode_length, beam_size,
#                                   top_beams, alpha, use_tpu)
#























































"""
**************************************************************************************************************************
**************************************************************************************************************************
Transformer Hparams!
**************************************************************************************************************************
**************************************************************************************************************************
"""









@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_2020debug_1():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")


  #hparams = transformer_small_local();

  #hparams = transformer_tiny_local();
  hparams = transformer_base_single_gpu_local()
  #hparams = transformer_big_single_gpu_local()
  #hparams = transformer_big_local()
  #hparams.set_hparam("max_length", 256)

  hparams.set_hparam("batch_size", 2048)

  hparams.set_hparam("hidden_size", 1024)
  hparams.set_hparam("filter_size", 1024)


  #hparams.set_hparam("ffn_layer", "conv_relu_conv")
  hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  hparams.add_hparam("conv_module_kernel_size", 1)
  #hparams.add_hparam("conv_module_kernel_size", 1)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128]])
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  #hparams.set_hparam("self_attention_type", "pointwise_attention")
  hparams.set_hparam("self_attention_type", "dot_product")

  #hparams.add_hparam("conv_padding", "SAME")
  hparams.add_hparam("conv_padding", "VALID")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  #hparams.set_hparam("pos", "none")  # timing, none
  hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64




  #  increase regularization
  hparams.add_hparam("attention_dropout", 0.5)

  hparams.add_hparam("relu_dropout", 0.5)



  return hparams











































#  FIXMETUNA:
@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_evolve14():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")


  #hparams = transformer_small_local();

  #hparams = transformer_tiny_local();
  hparams = transformer_base_single_gpu_local()
  #hparams = transformer_big_single_gpu_local()
  #hparams = transformer_big_local()
  #hparams.set_hparam("max_length", 256)

  hparams.set_hparam("batch_size", 2048)

  hparams.set_hparam("hidden_size", 1024)
  hparams.set_hparam("filter_size", 1024)


  hparams.set_hparam("ffn_layer", "conv_relu_conv")
  # hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  hparams.add_hparam("conv_module_kernel_size", 3)
  #hparams.add_hparam("conv_module_kernel_size", 1)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128]])
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  hparams.set_hparam("self_attention_type", "pointwise_attention")
  #hparams.set_hparam("self_attention_type", "dot_product")

  #hparams.add_hparam("conv_padding", "SAME")
  hparams.add_hparam("conv_padding", "VALID")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  hparams.set_hparam("pos", "none")  # timing, none
  #hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64





  #  need stronger regression Here
  hparams.layer_prepostprocess_dropout = 0.5
  hparams.attention_dropout = 0.5
  hparams.relu_dropout = 0.5


  return hparams














@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_evolve13():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")


  #hparams = transformer_small_local();

  #hparams = transformer_tiny_local();
  hparams = transformer_base_single_gpu_local()
  #hparams = transformer_big_single_gpu_local()
  #hparams = transformer_big_local()
  #hparams.set_hparam("max_length", 256)

  hparams.set_hparam("batch_size", 2048)

  hparams.set_hparam("hidden_size", 1024)
  hparams.set_hparam("filter_size", 1024)


  #hparams.set_hparam("ffn_layer", "conv_relu_conv")
  hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  #hparams.add_hparam("conv_module_kernel_size", 3)
  hparams.add_hparam("conv_module_kernel_size", 1)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128]])
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  #hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  #hparams.set_hparam("self_attention_type", "pointwise_attention")
  hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")
  #hparams.add_hparam("conv_padding", "VALID")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  #hparams.set_hparam("pos", "none")  # timing, none
  hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams














@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_evolve12():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")


  #hparams = transformer_small_local();

  #hparams = transformer_tiny_local();
  hparams = transformer_base_single_gpu_local()
  #hparams = transformer_big_single_gpu_local()
  #hparams = transformer_big_local()
  #hparams.set_hparam("max_length", 256)

  hparams.set_hparam("batch_size", 2048)

  hparams.set_hparam("hidden_size", 1024)
  hparams.set_hparam("filter_size", 1024)


  hparams.set_hparam("ffn_layer", "conv_relu_conv")
  #hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  #hparams.add_hparam("conv_module_kernel_size", 3)
  hparams.add_hparam("conv_module_kernel_size", 1)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128]])
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  #hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  #hparams.set_hparam("self_attention_type", "pointwise_attention")
  hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")
  #hparams.add_hparam("conv_padding", "VALID")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  #hparams.set_hparam("pos", "none")  # timing, none
  hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams


















@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_evolve11():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")


  #hparams = transformer_small_local();

  #hparams = transformer_tiny_local();
  hparams = transformer_base_single_gpu_local()
  #hparams = transformer_big_single_gpu_local()
  #hparams = transformer_big_local()
  #hparams.set_hparam("max_length", 256)

  hparams.set_hparam("batch_size", 2048)

  hparams.set_hparam("hidden_size", 1024)
  hparams.set_hparam("filter_size", 1024)


  hparams.set_hparam("ffn_layer", "conv_relu_conv")
  #hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  #hparams.add_hparam("conv_module_kernel_size", 3)
  hparams.add_hparam("conv_module_kernel_size", 1)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128]])
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  #hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  hparams.set_hparam("self_attention_type", "pointwise_attention")
  #hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")
  #hparams.add_hparam("conv_padding", "VALID")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  #hparams.set_hparam("pos", "none")  # timing, none
  hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams














































#  Experimetn HPARAMS:


@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_evolve9():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")


  #hparams = transformer_small_local();

  #hparams = transformer_tiny_local();
  #hparams = transformer_base_single_gpu_local()
  hparams = transformer_big_single_gpu_local()
  #hparams = transformer_big_local()
  #hparams.set_hparam("max_length", 256)

  hparams.set_hparam("batch_size", 2048)


  hparams.set_hparam("ffn_layer", "conv_relu_conv")
  #hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  hparams.add_hparam("conv_module_kernel_size", 3)
  #hparams.add_hparam("conv_module_kernel_size", 1)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128]])
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  #hparams.set_hparam("self_attention_type", "pointwise_attention")
  hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")
  #hparams.add_hparam("conv_padding", "VALID")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  #hparams.set_hparam("pos", "none")  # timing, none
  hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams





@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_evolve8():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")


  #hparams = transformer_small_local();

  #hparams = transformer_tiny_local();
  #hparams = transformer_base_single_gpu_local()
  hparams = transformer_big_single_gpu_local()
  #hparams = transformer_big_local()
  #hparams.set_hparam("max_length", 256)

  #hparams.set_hparam("batch_size", 1024)


  #hparams.set_hparam("ffn_layer", "conv_relu_conv")
  #hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  #hparams.add_hparam("conv_module_kernel_size", 3)
  #hparams.add_hparam("conv_module_kernel_size", 1)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128]])
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  #hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  #hparams.set_hparam("self_attention_type", "pointwise_attention")
  #hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")
  #hparams.add_hparam("conv_padding", "VALID")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  #hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  #hparams.set_hparam("pos", "none")  # timing, none
  hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams






@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_evolve7():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")


  #hparams = transformer_small_local();

  #hparams = transformer_tiny_local();
  #hparams = transformer_base_single_gpu_local()
  hparams = transformer_big_single_gpu_local()
  #hparams = transformer_big_local()
  #hparams.set_hparam("max_length", 256)

  #hparams.set_hparam("batch_size", 1024)


  #hparams.set_hparam("ffn_layer", "conv_relu_conv")
  hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  #hparams.add_hparam("conv_module_kernel_size", 3)
  hparams.add_hparam("conv_module_kernel_size", 1) #  FIXMETUNA:  only change


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128]])
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  #hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  #hparams.set_hparam("self_attention_type", "pointwise_attention")
  hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")
  #hparams.add_hparam("conv_padding", "VALID")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  #hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  #hparams.set_hparam("pos", "none")  # timing, none
  hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams








@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_evolve6():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")


  #hparams = transformer_small_local();

  #hparams = transformer_tiny_local();
  hparams = transformer_base_single_gpu_local()
  #hparams = transformer_big_single_gpu_local()
  #hparams = transformer_big_local()
  #hparams.set_hparam("max_length", 256)

  hparams.set_hparam("batch_size", 2048)

  hparams.set_hparam("hidden_size", 1024)
  hparams.set_hparam("filter_size", 1024)


  hparams.set_hparam("ffn_layer", "conv_relu_conv")
  #hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  hparams.add_hparam("conv_module_kernel_size", 3)
  #hparams.add_hparam("conv_module_kernel_size", 1)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128]])
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  #hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  hparams.set_hparam("self_attention_type", "pointwise_attention")
  #hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")
  #hparams.add_hparam("conv_padding", "VALID")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  #hparams.set_hparam("pos", "none")  # timing, none
  hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams











@registry.register_hparams
def conv_transformer_small_wmtende_v3():


  print("\n\n\n")
  print("")
  print("HPARAMS1!!")
  print("")
  print("\n\n\n")





  #hparams = transformer_tiny_local();
  hparams = transformer_small_local()

  hparams.set_hparam("max_length", 1024)


  hparams.set_hparam("ffn_layer", "conv_relu_conv")
  # hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  hparams.add_hparam("conv_module_kernel_size", 3)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  # hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  hparams.set_hparam("self_attention_type", "pointwise_attention")
  #  hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here
  hparams.set_hparam("pos", "none")  # timing, none
  # hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64

  #  #  TODO: convolutional transformer experiment hparams below
  #  #  TODO:  I'll have a set of if/else statements switching up the model
  #  #         on the fly above
  #  hparams.depthwise_seperable_convolutions = True
  #  hparams.add_positional_encodings = False

  #  #  attention type options: {"pointwise", "pointwise_w_final_dot_product_layer", "dot_product"}
  #  hparams.attention_type = "pointwise"

  #  #  output heads will probably need to involve the dataset and many
  #  #  additional hacks, I'll keep it as a hparam here for a future date
  #  #  output head options: {"seq", "class", "two_headed"}
  #  hparams.output_heads = "seq"

  return hparams














#  Experimetn HPARAMS:

@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_evolve2():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")


  #hparams = transformer_small_local();

  #hparams = transformer_tiny_local();
  hparams = transformer_base_single_gpu_local()
  #hparams = transformer_big_single_gpu_local()

  hparams.set_hparam("max_length", 256)



  hparams.set_hparam("ffn_layer", "conv_relu_conv")
  #hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  #hparams.add_hparam("conv_module_kernel_size", 3)
  hparams.add_hparam("conv_module_kernel_size", 3)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128],[1,2],[4,8],[16,32],[64,128]])
  #hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  #hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  hparams.set_hparam("self_attention_type", "pointwise_attention")
  #hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  hparams.set_hparam("pos", "none")  # timing, none
  #hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams




#  Experimetn HPARAMS:

@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1_evolve():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")



  #hparams = transformer_tiny_local();
  hparams = transformer_base_single_gpu_local()
  #hparams = transformer_big_single_gpu_local()

  hparams.set_hparam("hidden_size", 1024)
  hparams.set_hparam("filter_size", 1024)


  hparams.set_hparam("max_length", 1024)


  hparams.set_hparam("ffn_layer", "conv_relu_conv")
  #hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  hparams.add_hparam("conv_module_kernel_size", 3)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  # hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  hparams.set_hparam("self_attention_type", "pointwise_attention")
  #hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  hparams.set_hparam("pos", "none")  # timing, none


  # hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams



@registry.register_hparams
def conv_transformer_exp1_ctweqnumlayers1():


  print("\n\n\n")
  print("")
  print("HPARAMS2!!")
  print("")
  print("\n\n\n")



  #hparams = transformer_tiny_local();
  hparams = transformer_big_single_gpu_local()
  #hparams = transformer_base_single_gpu_local()

  hparams.set_hparam("max_length", 1024)


  hparams.set_hparam("ffn_layer", "conv_relu_conv")
  #hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  hparams.add_hparam("conv_module_kernel_size", 3)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  # hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  hparams.set_hparam("self_attention_type", "pointwise_attention")
  #hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here


  hparams.set_hparam("pos", "none")  # timing, none
  # hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams





#  Experimetn HPARAMS:

@registry.register_hparams
def conv_transformer_exp1_ctweqnumparams2():


  print("\n\n\n")
  print("")
  print("HPARAMS3!!")
  print("")
  print("\n\n\n")



  #hparams = transformer_tiny_local();
  hparams = transformer_base_single_gpu_local()

  #hparams.set_hparam("hidden_size", 512)
  hparams.set_hparam("hidden_size", 304)

  hparams.set_hparam("max_length", 1024)


  hparams.set_hparam("ffn_layer", "conv_relu_conv")
  # hparams.set_hparam("ffn_layer", "dense_relu_dense")
  # hparams.set_hparam("ffn_layer", "expdilconv_relu_expdilconv")

  #  NOTE: with the current implementation, if this is set to 1 then the input
  #  and output of V will become fully connected layers
  hparams.add_hparam("conv_module_kernel_size", 3)


  #  TODO:  currently restarts at layer 1 in the decoder, might want to make layer
  #  counting continue through the whole network without restarting?
  # hparams.add_hparam("conv_module_dilations", [[1,2],[4,8],[16,32],[64,128],[256,512],[1,2],[4,8],[16,32],[64,128],[256,512]])
  hparams.add_hparam("conv_module_dilations", [[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1],[1,1]])


  hparams.set_hparam("self_attention_type", "pointwise_attention")
  #  hparams.set_hparam("self_attention_type", "dot_product")

  hparams.add_hparam("conv_padding", "SAME")


  #  TODO:  this one is odd, list of which entire layers are depthwise separable
  #  vs normal convolutions, same as dilations, currently restarts at layer 1 for
  #  both the encoder and decoder
  #  FIXME:  currently on conv_relu_conv attention will use layer specific
  #  depthwise-sep convs, need to make this much more general for experiments 5
  #  FIXME: FIXME: FIXME:
  #  FIXME:
  #  FIXME:
  # hparams.add_hparam("depthwise_sep", False)
  # hparams.add_hparam("depthwise_sep", [False,True,True,True,True,True,True,True,True,True,True,True,True,True])
  hparams.add_hparam("depthwise_sep", [False,False,False,False,False,False,False,False,False,False,False,False,False,False])


  # #  FIXME:  this is a temporary hack, need to move this to the problem
  # #  FIXME:  or need to improve shaping information deep in the network somewhere
  # hparams.add_hparam("seq_len", 240)

  # #  this lets you set how large the hinorm_typedden filters in the module are
  #hparams.add_hparam("value_formation_module_hidden_dim", 512)

  #  FIXME:  default is layer, could experiment later, don't know how it works
  hparams.set_hparam("norm_type", "layer") # "batch", layer", "noam", "none"

  #  removing timing encoding here
  hparams.set_hparam("pos", "none")  # timing, none
  # hparams.set_hparam("pos", "timing")  # timing, none



  #  add/remove auto-regression here
  hparams.add_hparam("auto_regression", True)
  #  hparams.add_hparam("auto_regression", False)

  # hparams.use_fixed_batch_size = True
  # hparams.batch_size = 64


  return hparams



























#####################################################################################
#####################################################################################
#####################################################################################





@registry.register_hparams
def transformer_base_v1_local():
  """Set of hyperparameters."""
  hparams = common_hparams.basic_params1()
  hparams.norm_type = "layer"
  hparams.hidden_size = 512
  hparams.batch_size = 4096
  hparams.max_length = 256
  hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
  hparams.optimizer_adam_epsilon = 1e-9
  hparams.learning_rate_schedule = "legacy"
  hparams.learning_rate_decay_scheme = "noam"
  hparams.learning_rate = 0.1
  hparams.learning_rate_warmup_steps = 4000
  hparams.initializer_gain = 1.0
  hparams.num_hidden_layers = 6
  hparams.initializer = "uniform_unit_scaling"
  hparams.weight_decay = 0.0
  hparams.optimizer_adam_beta1 = 0.9
  hparams.optimizer_adam_beta2 = 0.98
  hparams.num_sampled_classes = 0
  hparams.label_smoothing = 0.1
  hparams.shared_embedding_and_softmax_weights = True
  hparams.symbol_modality_num_shards = 16

  # Add new ones like this.
  hparams.add_hparam("filter_size", 2048)
  # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
  hparams.add_hparam("num_encoder_layers", 0)
  hparams.add_hparam("num_decoder_layers", 0)
  # Attention-related flags.
  hparams.add_hparam("num_heads", 8)
  hparams.add_hparam("attention_key_channels", 0)
  hparams.add_hparam("attention_value_channels", 0)
  hparams.add_hparam("ffn_layer", "dense_relu_dense")
  hparams.add_hparam("parameter_attention_key_channels", 0)
  hparams.add_hparam("parameter_attention_value_channels", 0)
  # All hyperparameters ending in "dropout" are automatically set to 0.0
  # when not in training mode.
  hparams.add_hparam("attention_dropout", 0.0)
  hparams.add_hparam("attention_dropout_broadcast_dims", "")
  hparams.add_hparam("relu_dropout", 0.0)
  hparams.add_hparam("relu_dropout_broadcast_dims", "")
  hparams.add_hparam("pos", "timing")  # timing, none
  hparams.add_hparam("nbr_decoder_problems", 1)
  hparams.add_hparam("proximity_bias", False)
  hparams.add_hparam("causal_decoder_self_attention", True)
  hparams.add_hparam("use_pad_remover", True)
  hparams.add_hparam("self_attention_type", "dot_product")
  hparams.add_hparam("conv_first_kernel", 3)
  hparams.add_hparam("attention_variables_3d", False)
  hparams.add_hparam("use_target_space_embedding", True)
  # These parameters are only used when ffn_layer=="local_moe_tpu"
  hparams.add_hparam("moe_overhead_train", 1.0)
  hparams.add_hparam("moe_overhead_eval", 2.0)
  hparams.moe_num_experts = 16
  hparams.moe_loss_coef = 1e-3
  # If specified, use this value instead of problem name in metrics.py.
  # This is useful for programs that can automatically compare experiments side
  #   by side based on the same metric names.
  hparams.add_hparam("overload_eval_metric_name", "")
  # For making a transformer encoder unidirectional by using masked
  # attention.
  hparams.add_hparam("unidirectional_encoder", False)
  # For hard attention.
  hparams.add_hparam("hard_attention_k", 0)
  return hparams


@registry.register_hparams
def transformer_base_v2_local():
  """Set of hyperparameters."""
  hparams = transformer_base_v1_local()
  hparams.layer_preprocess_sequence = "n"
  hparams.layer_postprocess_sequence = "da"
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.attention_dropout = 0.1
  hparams.relu_dropout = 0.1
  hparams.learning_rate_warmup_steps = 8000
  hparams.learning_rate = 0.2
  return hparams




@registry.register_hparams
def transformer_base_v3_local():
  """Base parameters for Transformer model."""
  # Update parameters here, then occasionally cut a versioned set, e.g.
  # transformer_base_v2.
  hparams = transformer_base_v2_local()
  hparams.optimizer_adam_beta2 = 0.997
  # New way of specifying learning rate schedule.
  # Equivalent to previous version.
  hparams.learning_rate_schedule = (
      "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
  hparams.learning_rate_constant = 2.0
  return hparams


@registry.register_hparams
def transformer_base_local():
  """Base parameters for Transformer model."""
  hparams = transformer_base_v3_local()
  return hparams



@registry.register_hparams
def transformer_small_local():
  hparams = transformer_base_local()
  hparams.num_hidden_layers = 2
  hparams.hidden_size = 256
  hparams.filter_size = 1024
  hparams.num_heads = 4
  return hparams


@registry.register_hparams
def transformer_tiny_local():
  hparams = transformer_base_local()
  hparams.num_hidden_layers = 1
  hparams.hidden_size = 64
  hparams.filter_size = 256
  hparams.num_heads = 4
  return hparams



























# @registry.register_hparams
# def transformer_base_v1_local():
#   """Set of hyperparameters."""
#   hparams = common_hparams.basic_params1()
#   hparams.norm_type = "layer"
#   hparams.hidden_size = 512
#   hparams.batch_size = 4096
#   hparams.max_length = 256
#   hparams.clip_grad_norm = 0.  # i.e. no gradient clipping
#   hparams.optimizer_adam_epsilon = 1e-9
#   hparams.learning_rate_schedule = "legacy"
#   hparams.learning_rate_decay_scheme = "noam"
#   hparams.learning_rate = 0.1
#   hparams.learning_rate_warmup_steps = 4000
#   hparams.initializer_gain = 1.0
#   hparams.num_hidden_layers = 6
#   hparams.initializer = "uniform_unit_scaling"
#   hparams.weight_decay = 0.0
#   hparams.optimizer_adam_beta1 = 0.9
#   hparams.optimizer_adam_beta2 = 0.98
#   hparams.num_sampled_classes = 0
#   hparams.label_smoothing = 0.1
#   hparams.shared_embedding_and_softmax_weights = True
#   hparams.symbol_modality_num_shards = 16
#
#   # Add new ones like this.
#   hparams.add_hparam("filter_size", 2048)
#   # Layer-related flags. If zero, these fall back on hparams.num_hidden_layers.
#   hparams.add_hparam("num_encoder_layers", 0)
#   hparams.add_hparam("num_decoder_layers", 0)
#   # Attention-related flags.
#   hparams.add_hparam("num_heads", 8)
#   hparams.add_hparam("attention_key_channels", 0)
#   hparams.add_hparam("attention_value_channels", 0)
#   hparams.add_hparam("ffn_layer", "dense_relu_dense")
#   hparams.add_hparam("parameter_attention_key_channels", 0)
#   hparams.add_hparam("parameter_attention_value_channels", 0)
#   # All hyperparameters ending in "dropout" are automatically set to 0.0
#   # when not in training mode.
#   hparams.add_hparam("attention_dropout", 0.0)
#   hparams.add_hparam("attention_dropout_broadcast_dims", "")
#   hparams.add_hparam("relu_dropout", 0.0)
#   hparams.add_hparam("relu_dropout_broadcast_dims", "")
#   hparams.add_hparam("pos", "timing")  # timing, none
#   hparams.add_hparam("nbr_decoder_problems", 1)
#   hparams.add_hparam("proximity_bias", False)
#   hparams.add_hparam("causal_decoder_self_attention", True)
#   hparams.add_hparam("use_pad_remover", True)
#   hparams.add_hparam("self_attention_type", "dot_product")
#   hparams.add_hparam("conv_first_kernel", 3)
#   hparams.add_hparam("attention_variables_3d", False)
#   hparams.add_hparam("use_target_space_embedding", True)
#   # These parameters are only used when ffn_layer=="local_moe_tpu"
#   hparams.add_hparam("moe_overhead_train", 1.0)
#   hparams.add_hparam("moe_overhead_eval", 2.0)
#   hparams.moe_num_experts = 16
#   hparams.moe_loss_coef = 1e-3
#   # If specified, use this value instead of problem name in metrics.py.
#   # This is useful for programs that can automatically compare experiments side
#   #   by side based on the same metric names.
#   hparams.add_hparam("overload_eval_metric_name", "")
#   # For making a transformer encoder unidirectional by using masked
#   # attention.
#   hparams.add_hparam("unidirectional_encoder", False)
#   # For hard attention.
#   hparams.add_hparam("hard_attention_k", 0)
#   return hparams
#
#
# @registry.register_hparams
# def transformer_base_v2_local():
#   """Set of hyperparameters."""
#   hparams = transformer_base_v1_local()
#   hparams.layer_preprocess_sequence = "n"
#   hparams.layer_postprocess_sequence = "da"
#   hparams.layer_prepostprocess_dropout = 0.1
#   hparams.attention_dropout = 0.1
#   hparams.relu_dropout = 0.1
#   hparams.learning_rate_warmup_steps = 8000
#   hparams.learning_rate = 0.2
#   return hparams
#
# #
# # @registry.register_hparams
# # def transformer_base_vq_ada_32ex_packed():
# #   """Set of hyperparameters for lm1b packed following tpu params."""
# #   hparams = transformer_base_v2()
# #   expert_utils.update_hparams_for_vq_gating(hparams)
# #   hparams.moe_num_experts = 32
# #   hparams.gating_type = "vq"
# #   # this gives us a batch size of 16 because each seq is len 256
# #   hparams.batch_size = 5072
# #   hparams.ffn_layer = "local_moe"
# #   hparams.shared_embedding_and_softmax_weights = False
# #   hparams.learning_rate_warmup_steps = 10000
# #   # one epoch for languagemodel_lm1b32k_packed = 27200 steps w/ bsize 128
# #   hparams.learning_rate_decay_steps = 27200
# #   hparams.num_heads = 4
# #   hparams.num_blocks = 1
# #   hparams.moe_k = 1
# #   hparams.num_decoder_layers = 6
# #   hparams.label_smoothing = 0.
# #   hparams.layer_prepostprocess_dropout = 0.1
# #   hparams.layer_postprocess_sequence = "dan"
# #   hparams.layer_preprocess_sequence = "none"
# #   hparams.weight_decay = 1e-06
# #   hparams.attention_dropout = 0.1
# #   hparams.optimizer = "Adafactor"
# #   hparams.learning_rate_schedule = "linear_warmup*rsqrt_decay*linear_decay"
# #   hparams.activation_dtype = "float32"
# #   hparams.learning_rate = 0.1
# #   hparams.learning_rate_constant = 1.0
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_topk_16_packed():
# #   hparams = transformer_base_vq_ada_32ex_packed()
# #   hparams.gating_type = "topk"
# #   hparams.moe_num_experts = 16
# #   hparams.moe_k = 2
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_base_vq1_16_nb1_packed_nda_b01_scales():
# #   """Set of hyperparameters."""
# #   hparams = transformer_base_vq_ada_32ex_packed()
# #   hparams.use_scales = int(True)
# #   hparams.moe_num_experts = 16
# #   hparams.moe_k = 1
# #   hparams.beta = 0.1
# #   hparams.layer_preprocess_sequence = "n"
# #   hparams.layer_postprocess_sequence = "da"
# #   hparams.ema = False
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_base_vq1_16_nb1_packed_dan_b01_scales():
# #   """Set of hyperparameters."""
# #   hparams = transformer_base_vq_ada_32ex_packed()
# #   hparams.use_scales = int(True)
# #   hparams.moe_num_experts = 16
# #   hparams.moe_k = 1
# #   hparams.beta = 0.1
# #   hparams.ema = False
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_base_vq1_16_nb1_packed_nda_b01_scales_dialog():
# #   """Set of hyperparameters."""
# #   hparams = transformer_base_vq1_16_nb1_packed_nda_b01_scales()
# #   hparams.batch_size = 2048
# #   hparams.max_length = 1024
# #   hparams.filter_size = 3072
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_ada_lmpackedbase():
# #   """Set of hyperparameters."""
# #   hparams = transformer_base_vq_ada_32ex_packed()
# #   hparams.ffn_layer = "dense_relu_dense"
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_ada_lmpackedbase_dialog():
# #   """Set of hyperparameters."""
# #   hparams = transformer_base_vq_ada_32ex_packed()
# #   hparams.max_length = 1024
# #   hparams.ffn_layer = "dense_relu_dense"
# #   hparams.batch_size = 4096
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_ada_lmpackedbase_relative():
# #   """Set of hyperparameters."""
# #   hparams = transformer_base_vq_ada_32ex_packed()
# #   hparams.ffn_layer = "dense_relu_dense"
# #   return hparams
#
#
# @registry.register_hparams
# def transformer_base_v3_local():
#   """Base parameters for Transformer model."""
#   # Update parameters here, then occasionally cut a versioned set, e.g.
#   # transformer_base_v2.
#   hparams = transformer_base_v2_local()
#   hparams.optimizer_adam_beta2 = 0.997
#   # New way of specifying learning rate schedule.
#   # Equivalent to previous version.
#   hparams.learning_rate_schedule = (
#       "constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size")
#   hparams.learning_rate_constant = 2.0
#   return hparams
#
#
# @registry.register_hparams
# def transformer_base_local():
#   """Base parameters for Transformer model."""
#   hparams = transformer_base_v3_local()
#   return hparams
#
# #


@registry.register_hparams
def transformer_big_local():
  """HParams for transformer big model on WMT."""
  hparams = transformer_base_local()
  hparams.hidden_size = 1024
  hparams.filter_size = 4096
  # Reduce batch size to 2048 from 4096 to be able to train the model on a GPU
  # with 12 GB memory. For example, NVIDIA TITAN V GPU.
  hparams.batch_size = 2048
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams

@registry.register_hparams
def transformer_big_single_gpu_local_adjusted_batch():
  """HParams for transformer big model on WMT."""
  hparams = transformer_big_local()
  hparams.batch_size = 4096
  return hparams

# #
# # @registry.register_hparams
# # def transformer_tall():
# #   """Hparams for transformer on LM for pretraining/finetuning/mixing."""
# #   hparams = transformer_base()
# #   hparams.batch_size = 2048
# #   hparams.hidden_size = 768
# #   hparams.filter_size = 3072
# #   hparams.num_hidden_layers = 12
# #   hparams.num_heads = 12
# #   hparams.label_smoothing = 0.0
# #   hparams.max_length = 1024
# #   hparams.eval_drop_long_sequences = True
# #   hparams.multiproblem_mixing_schedule = "pretrain"
# #   hparams.multiproblem_vocab_size = 65536
# #   hparams.clip_grad_norm = 1.0
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tall_finetune_tied():
# #   """Tied means fine-tune CNN/DM summarization as LM."""
# #   hparams = transformer_tall()
# #   hparams.multiproblem_max_input_length = 750
# #   hparams.multiproblem_max_target_length = 100
# #   hparams.multiproblem_schedule_max_examples = 0
# #   hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
# #   hparams.learning_rate_constant = 5e-5
# #   hparams.learning_rate_warmup_steps = 100
# #   # Set train steps to learning_rate_decay_steps or less
# #   hparams.learning_rate_decay_steps = 80000
# #   hparams.multiproblem_target_eval_only = True
# #   hparams.multiproblem_reweight_label_loss = True
# #   hparams.multiproblem_label_weight = 1.0
# #   hparams.optimizer = "true_adam"
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tall_train_tied():
# #   """Tied means train CNN/DM summarization as LM."""
# #   hparams = transformer_tall()
# #   hparams.multiproblem_max_input_length = 750
# #   hparams.multiproblem_max_target_length = 100
# #   hparams.multiproblem_schedule_max_examples = 0
# #   hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
# #   hparams.learning_rate_constant = 2e-4
# #   hparams.learning_rate_warmup_steps = 8000
# #   # Set train steps to learning_rate_decay_steps or less
# #   hparams.learning_rate_decay_steps = 150000
# #   hparams.multiproblem_target_eval_only = True
# #   hparams.multiproblem_reweight_label_loss = True
# #   hparams.multiproblem_label_weight = 1.0
# #   hparams.optimizer = "true_adam"
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tall_finetune_uniencdec():
# #   """Fine-tune CNN/DM with a unidirectional encoder and decoder."""
# #   hparams = transformer_tall()
# #   hparams.max_input_seq_length = 750
# #   hparams.max_target_seq_length = 100
# #   hparams.optimizer = "true_adam"
# #   hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
# #   hparams.learning_rate_decay_steps = 80000
# #   hparams.learning_rate_constant = 5e-5
# #   hparams.learning_rate_warmup_steps = 100
# #   hparams.unidirectional_encoder = True
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tall_train_uniencdec():
# #   """Train CNN/DM with a unidirectional encoder and decoder."""
# #   hparams = transformer_tall()
# #   hparams.max_input_seq_length = 750
# #   hparams.max_target_seq_length = 100
# #   hparams.optimizer = "true_adam"
# #   hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
# #   hparams.learning_rate_decay_steps = 150000
# #   hparams.learning_rate_constant = 2e-4
# #   hparams.unidirectional_encoder = True
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tall_finetune_textclass():
# #   """Hparams for transformer on LM for finetuning on text class problems."""
# #   hparams = transformer_tall()
# #   hparams.learning_rate_constant = 6.25e-5
# #   hparams.learning_rate_schedule = ("linear_warmup*constant*linear_decay")
# #   hparams.multiproblem_schedule_max_examples = 0
# #   hparams.multiproblem_target_eval_only = True
# #   hparams.learning_rate_warmup_steps = 50
# #   # Set train steps to learning_rate_decay_steps or less
# #   hparams.learning_rate_decay_steps = 25000
# #   hparams.multiproblem_reweight_label_loss = True
# #   hparams.multiproblem_label_weight = 0.95
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tall_pretrain_lm():
# #   """Hparams for transformer on LM pretraining (with 64k vocab)."""
# #   hparams = transformer_tall()
# #   hparams.learning_rate_constant = 2e-4
# #   hparams.learning_rate_schedule = ("linear_warmup*constant*cosdecay")
# #   hparams.optimizer = "adam_w"
# #   hparams.optimizer_adam_beta1 = 0.9
# #   hparams.optimizer_adam_beta2 = 0.999
# #   hparams.optimizer_adam_epsilon = 1e-8
# #   # Set max examples to something big when pretraining only the LM, definitely
# #   # something an order of magnitude bigger than number of train steps.
# #   hparams.multiproblem_schedule_max_examples = 5e8
# #   # Set train steps to learning_rate_decay_steps or less
# #   hparams.learning_rate_decay_steps = 5000000
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tall_pretrain_lm_tpu_adafactor():
# #   """Hparams for transformer on LM pretraining (with 64k vocab) on TPU."""
# #   hparams = transformer_tall_pretrain_lm()
# #   update_hparams_for_tpu(hparams)
# #   hparams.max_length = 1024
# #   # For multi-problem on TPU we need it in absolute examples.
# #   hparams.batch_size = 8
# #   hparams.multiproblem_vocab_size = 2**16
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tall_pretrain_lm_tpu_adafactor_large():
# #   """Hparams for transformer on LM pretraining on TPU, large model."""
# #   hparams = transformer_tall_pretrain_lm_tpu_adafactor()
# #   hparams.hidden_size = 1024
# #   hparams.num_heads = 16
# #   hparams.filter_size = 32768  # max fitting in 16G memory is 49152, batch 2
# #   hparams.batch_size = 4
# #   hparams.multiproblem_mixing_schedule = "constant"
# #   # Task order: lm/en-de/en-fr/en-ro/de-en/fr-en/ro-en/cnndm/mnli/squad.
# #   hparams.multiproblem_per_task_threshold = "320,80,160,1,80,160,2,20,10,5"
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tall_pretrain_lm_tpu():
# #   """Hparams for transformer on LM pretraining on TPU with AdamW."""
# #   hparams = transformer_tall_pretrain_lm_tpu_adafactor()
# #   # Optimizer gets reset in update_hparams_for_tpu so we set it again here.
# #   hparams.learning_rate_constant = 2e-4
# #   hparams.learning_rate_schedule = ("linear_warmup * constant * cosdecay")
# #   hparams.optimizer = "adam_w"
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tall_big():
# #   """Hparams for transformer on LM+MNLI."""
# #   hparams = transformer_tall()
# #   hparams.num_hidden_layers = 18
# #   return hparams
# #
# #
@registry.register_hparams
def transformer_big_single_gpu_local():
  """HParams for transformer big model for single GPU."""
  hparams = transformer_big_local()
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate_warmup_steps = 16000
  return hparams


@registry.register_hparams
def transformer_base_single_gpu_local():
  """HParams for transformer base model for single GPU."""
  hparams = transformer_base_local()
  hparams.batch_size = 1024
  hparams.learning_rate_schedule = "constant*linear_warmup*rsqrt_decay"
  hparams.learning_rate_constant = 0.1
  hparams.learning_rate_warmup_steps = 16000
  return hparams

@registry.register_hparams
def transformer_base_single_gpu_local_adjusted_batch():
  """HParams for transformer base model for single GPU."""
  hparams = transformer_base_local()
  hparams.batch_size = 4096
  hparams.learning_rate_schedule = "constant*linear_warmup*rsqrt_decay"
  hparams.learning_rate_constant = 0.1
  hparams.learning_rate_warmup_steps = 16000
  return hparams
# #
# # @registry.register_hparams
# # def transformer_base_multistep8():
# #   """HParams for simulating 8 GPUs with MultistepAdam optimizer."""
# #   hparams = transformer_base()
# #   hparams.optimizer = "MultistepAdam"
# #   hparams.optimizer_multistep_accumulate_steps = 8
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_parsing_base():
# #   """HParams for parsing on WSJ only."""
# #   hparams = transformer_base()
# #   hparams.attention_dropout = 0.2
# #   hparams.layer_prepostprocess_dropout = 0.2
# #   hparams.max_length = 512
# #   hparams.learning_rate_warmup_steps = 16000
# #   hparams.hidden_size = 1024
# #   hparams.learning_rate = 0.05
# #   hparams.shared_embedding_and_softmax_weights = False
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_parsing_big():
# #   """HParams for parsing on WSJ semi-supervised."""
# #   hparams = transformer_big()
# #   hparams.max_length = 512
# #   hparams.shared_source_target_embedding = False
# #   hparams.learning_rate_warmup_steps = 4000
# #   hparams.layer_prepostprocess_dropout = 0.1
# #   hparams.batch_size = 2048
# #   hparams.learning_rate = 0.05
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_parsing_ice():
# #   """HParams for parsing and tagging Icelandic text."""
# #   hparams = transformer_base_single_gpu()
# #   hparams.batch_size = 4096
# #   hparams.shared_embedding_and_softmax_weights = False
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tiny():
# #   hparams = transformer_base()
# #   hparams.num_hidden_layers = 2
# #   hparams.hidden_size = 128
# #   hparams.filter_size = 512
# #   hparams.num_heads = 4
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_test():
# #   hparams = transformer_base()
# #   hparams.num_hidden_layers = 2
# #   hparams.hidden_size = 16
# #   hparams.filter_size = 8
# #   hparams.num_heads = 2
# #   return hparams
#
#
# @registry.register_hparams
# def transformer_small_local():
#   hparams = transformer_base_local()
#   hparams.num_hidden_layers = 2
#   hparams.hidden_size = 256
#   hparams.filter_size = 1024
#   hparams.num_heads = 4
#   return hparams
#
# #
# # @registry.register_hparams
# # def transformer_l2():
# #   hparams = transformer_base()
# #   hparams.num_hidden_layers = 2
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_l4():
# #   hparams = transformer_base()
# #   hparams.num_hidden_layers = 4
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_l8():
# #   hparams = transformer_base()
# #   hparams.num_hidden_layers = 8
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_l10():
# #   hparams = transformer_base()
# #   hparams.num_hidden_layers = 10
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_h1():
# #   hparams = transformer_base()
# #   hparams.num_heads = 1
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_h4():
# #   hparams = transformer_base()
# #   hparams.num_heads = 4
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_h16():
# #   hparams = transformer_base()
# #   hparams.num_heads = 16
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_h32():
# #   hparams = transformer_base()
# #   hparams.num_heads = 32
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_k128():
# #   hparams = transformer_base()
# #   hparams.attention_key_channels = 128
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_k256():
# #   hparams = transformer_base()
# #   hparams.attention_key_channels = 256
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_ff1024():
# #   hparams = transformer_base()
# #   hparams.filter_size = 1024
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_ff4096():
# #   hparams = transformer_base()
# #   hparams.filter_size = 4096
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_dr0():
# #   hparams = transformer_base()
# #   hparams.layer_prepostprocess_dropout = 0.0
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_dr2():
# #   hparams = transformer_base()
# #   hparams.layer_prepostprocess_dropout = 0.2
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_ls0():
# #   hparams = transformer_base()
# #   hparams.label_smoothing = 0.0
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_ls2():
# #   hparams = transformer_base()
# #   hparams.label_smoothing = 0.2
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_hs256():
# #   hparams = transformer_base()
# #   hparams.hidden_size = 256
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_hs1024():
# #   hparams = transformer_base()
# #   hparams.hidden_size = 1024
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_big_dr1():
# #   hparams = transformer_base()
# #   hparams.hidden_size = 1024
# #   hparams.filter_size = 4096
# #   hparams.num_heads = 16
# #   hparams.layer_prepostprocess_dropout = 0.1
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_big_enfr():
# #   hparams = transformer_big_dr1()
# #   hparams.shared_embedding_and_softmax_weights = False
# #   hparams.filter_size = 8192
# #   hparams.layer_prepostprocess_dropout = 0.1
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_big_enfr_tpu():
# #   hparams = transformer_big_enfr()
# #   # For performance, use fewer heads so that matrix dimensions are at least 128
# #   hparams.num_heads = 8
# #   update_hparams_for_tpu(hparams)
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_big_dr2():
# #   hparams = transformer_big_dr1()
# #   hparams.layer_prepostprocess_dropout = 0.2
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_parameter_attention_a():
# #   hparams = transformer_base()
# #   hparams.ffn_layer = "parameter_attention"
# #   hparams.filter_size = 1536
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_parameter_attention_b():
# #   hparams = transformer_base()
# #   hparams.ffn_layer = "parameter_attention"
# #   hparams.filter_size = 512
# #   hparams.parameter_attention_key_channels = 1024
# #   hparams.parameter_attention_value_channels = 1024
# #   hparams.num_heads = 16
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_prepend_v2():
# #   hparams = transformer_base_v2()
# #   hparams.prepend_mode = "prepend_inputs_masked_attention"
# #   hparams.max_length = 0
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_prepend_v1():
# #   hparams = transformer_base_v1()
# #   hparams.prepend_mode = "prepend_inputs_masked_attention"
# #   hparams.max_length = 0
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_prepend():
# #   return transformer_prepend_v2()
# #
# #
# # @registry.register_ranged_hparams
# # def transformer_base_range(rhp):
# #   """Small range of hyperparameters."""
# #   # After starting from base, set intervals for some parameters.
# #   rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
# #   rhp.set_discrete("learning_rate_warmup_steps",
# #                    [1000, 2000, 4000, 8000, 16000])
# #   rhp.set_float("initializer_gain", 0.5, 2.0)
# #   rhp.set_float("optimizer_adam_beta1", 0.85, 0.95)
# #   rhp.set_float("optimizer_adam_beta2", 0.97, 0.99)
# #   rhp.set_float("weight_decay", 0.0, 1e-4)
# #
# #
# # @registry.register_hparams
# # def transformer_relative():
# #   """Use relative position embeddings instead of absolute position encodings."""
# #   hparams = transformer_base()
# #   hparams.pos = None
# #   hparams.self_attention_type = "dot_product_relative"
# #   hparams.max_relative_position = 20
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_relative_tiny():
# #   hparams = transformer_relative()
# #   hparams.num_hidden_layers = 2
# #   hparams.hidden_size = 128
# #   hparams.filter_size = 512
# #   hparams.num_heads = 4
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_relative_big():
# #   hparams = transformer_big()
# #   hparams.pos = None
# #   hparams.self_attention_type = "dot_product_relative"
# #   hparams.max_relative_position = 20
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_timeseries():
# #   hparams = transformer_small()
# #   hparams.batch_size = 256
# #   hparams.learning_rate_warmup_steps = 2000
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_mlperf_tpu():
# #   """HParams for Transformer model on TPU for MLPerf on TPU 2x2."""
# #   hparams = transformer_base_v3()
# #   hparams.mlperf_mode = True
# #   hparams.symbol_modality_num_shards = 1
# #   hparams.max_length = 256  # ignored when using "_packed" problems
# #   hparams.batch_size = 2048  # per-chip batch size matches the reference model
# #   hparams.hidden_size = 1024
# #   hparams.filter_size = 4096
# #   hparams.num_heads = 16
# #   hparams.attention_dropout_broadcast_dims = "0,1"  # batch, heads
# #   hparams.relu_dropout_broadcast_dims = "1"  # length
# #   hparams.layer_prepostprocess_dropout_broadcast_dims = "1"  # length
# #   return hparams
# #
# #
# # def update_hparams_for_tpu(hparams):
# #   """Change hparams to be compatible with TPU training."""
# #
# #   # Adafactor uses less memory than Adam.
# #   # switch to Adafactor with its recommended learning rate scheme.
# #   hparams.optimizer = "Adafactor"
# #   hparams.learning_rate_schedule = "rsqrt_decay"
# #   hparams.learning_rate_warmup_steps = 10000
# #
# #   # Avoid an expensive concat on TPU.
# #   # >1 shards helps with faster parameter distribution on multi-GPU machines
# #   hparams.symbol_modality_num_shards = 1
# #
# #   # Adaptive batch sizes and sequence lengths are not supported on TPU.
# #   # Instead, every batch has the same sequence length and the same batch size.
# #   # Longer sequences are dropped and shorter ones are padded.
# #   #
# #   # It is therefore suggested to use a problem where examples have been combined
# #   # to a longer length, e.g. the "_packed" problems.
# #   #
# #   # For problems with variable sequence lengths, this parameter controls the
# #   # maximum sequence length.  Shorter sequences are dropped and longer ones
# #   # are padded.
# #   #
# #   # For problems with fixed sequence lengths - e.g. the "_packed" problems,
# #   # this hyperparameter is ignored.
# #   hparams.max_length = 64
# #
# #   # TPUs have less memory than GPUs, so decrease the batch size
# #   hparams.batch_size = 2048
# #
# #   # Using noise broadcast in the dropout layers saves memory during training.
# #   hparams.attention_dropout_broadcast_dims = "0,1"  # batch, heads
# #   hparams.relu_dropout_broadcast_dims = "1"  # length
# #   hparams.layer_prepostprocess_dropout_broadcast_dims = "1"  # length
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tpu():
# #   """HParams for Transformer model on TPU."""
# #   hparams = transformer_base()
# #   update_hparams_for_tpu(hparams)
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_timeseries_tpu():
# #   """HParams for running Transformer model on timeseries on TPU."""
# #   hparams = transformer_timeseries()
# #   update_hparams_for_tpu(hparams)
# #   hparams.batch_size = 256  # revert to value set in transformer_timeseries
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tpu_bf16_activation():
# #   """HParams for Transformer model with BF16 activation on TPU."""
# #   hparams = transformer_tpu()
# #   hparams.activation_dtype = "bfloat16"
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_fairseq_fp16_activation_big():
# #   """Hparams intended to mirror those used in arxiv.org/pdf/1806.00187.pdf."""
# #   hparams = transformer_big()
# #   hparams.activation_dtype = "float16"
# #   hparams.batch_size = 3584
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_packed_tpu():
# #   """Deprecated alias for transformer_tpu()."""
# #   return transformer_tpu()
# #
# #
# # @registry.register_hparams
# # def transformer_big_tpu():
# #   hparams = transformer_big()
# #   update_hparams_for_tpu(hparams)
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tiny_tpu():
# #   hparams = transformer_tiny()
# #   update_hparams_for_tpu(hparams)
# #   return hparams
# #
# #
# # @registry.register_ranged_hparams
# # def transformer_tiny_tpu_range(rhp):
# #   """Small range of hyperparameters."""
# #   rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
# #   rhp.set_float("weight_decay", 0.0, 2.0)
# #
# #
# # @registry.register_ranged_hparams
# # def transformer_tpu_range(rhp):
# #   """Small range of hyperparameters."""
# #   # After starting from base, set intervals for some parameters.
# #   rhp.set_float("learning_rate", 0.3, 3.0, scale=rhp.LOG_SCALE)
# #   rhp.set_discrete("learning_rate_warmup_steps",
# #                    [1000, 2000, 4000, 8000, 16000])
# #   rhp.set_float("initializer_gain", 0.5, 2.0)
# #   rhp.set_float("optimizer_adam_beta1", 0.85, 0.95)
# #   rhp.set_float("optimizer_adam_beta2", 0.97, 0.99)
# #   rhp.set_float("weight_decay", 0.0, 2.0)
# #
# #
# # @registry.register_hparams
# # def transformer_small_tpu():
# #   """TPU-friendly version of transformer_small.
# #
# #   Returns:
# #     an hparams object.
# #   """
# #   hparams = transformer_small()
# #   update_hparams_for_tpu(hparams)
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_clean():
# #   """No dropout, label smoothing, max_length."""
# #   hparams = transformer_base_v2()
# #   hparams.label_smoothing = 0.0
# #   hparams.layer_prepostprocess_dropout = 0.0
# #   hparams.attention_dropout = 0.0
# #   hparams.relu_dropout = 0.0
# #   hparams.max_length = 0
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_clean_big():
# #   hparams = transformer_clean()
# #   hparams.hidden_size = 1024
# #   hparams.filter_size = 4096
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_clean_big_tpu():
# #   hparams = transformer_clean_big()
# #   update_hparams_for_tpu(hparams)
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tpu_with_conv():
# #   """Cut down on the number of heads, and use convs instead."""
# #   hparams = transformer_tpu()
# #   hparams.num_heads = 4  # Heads are expensive on TPUs.
# #   hparams.ffn_layer = "conv_relu_conv"
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_lm_tpu_0():
# #   """HParams for training languagemodel_lm1b8k on tpu.  92M Params."""
# #   hparams = transformer_clean_big()
# #   update_hparams_for_tpu(hparams)
# #   hparams.num_heads = 4  # Heads are expensive on TPUs.
# #   hparams.batch_size = 4096
# #   hparams.shared_embedding_and_softmax_weights = False
# #   hparams.layer_prepostprocess_dropout = 0.1
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_lm_tpu_1():
# #   """HParams for training languagemodel_lm1b8k on tpu.  335M Params."""
# #   hparams = transformer_lm_tpu_0()
# #   hparams.hidden_size = 2048
# #   hparams.filter_size = 8192
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_librispeech_v1():
# #   """HParams for training ASR model on LibriSpeech V1."""
# #   hparams = transformer_base()
# #
# #   hparams.num_heads = 4
# #   hparams.filter_size = 1024
# #   hparams.hidden_size = 256
# #   hparams.num_encoder_layers = 5
# #   hparams.num_decoder_layers = 3
# #   hparams.learning_rate = 0.15
# #   hparams.batch_size = 6000000
# #
# #   librispeech.set_librispeech_length_hparams(hparams)
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_librispeech_v2():
# #   """HParams for training ASR model on LibriSpeech V2."""
# #   hparams = transformer_base()
# #
# #   hparams.max_length = 1240000
# #   hparams.max_input_seq_length = 1550
# #   hparams.max_target_seq_length = 350
# #   hparams.batch_size = 16
# #   hparams.num_decoder_layers = 4
# #   hparams.num_encoder_layers = 6
# #   hparams.hidden_size = 384
# #   hparams.learning_rate = 0.15
# #   hparams.daisy_chain_variables = False
# #   hparams.filter_size = 1536
# #   hparams.num_heads = 2
# #   hparams.ffn_layer = "conv_relu_conv"
# #   hparams.conv_first_kernel = 9
# #   hparams.weight_decay = 0
# #   hparams.layer_prepostprocess_dropout = 0.2
# #   hparams.relu_dropout = 0.2
# #
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_librispeech_tpu_v1():
# #   """HParams for training ASR model on Librispeech on TPU v1."""
# #   hparams = transformer_librispeech_v1()
# #   update_hparams_for_tpu(hparams)
# #
# #   hparams.batch_size = 16
# #   librispeech.set_librispeech_length_hparams(hparams)
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_librispeech_tpu_v2():
# #   """HParams for training ASR model on Librispeech on TPU v2."""
# #   hparams = transformer_librispeech_v2()
# #   update_hparams_for_tpu(hparams)
# #
# #   hparams.batch_size = 16
# #   librispeech.set_librispeech_length_hparams(hparams)
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_librispeech():
# #   """HParams for training ASR model on Librispeech."""
# #   return transformer_librispeech_v2()
# #
# #
# # @registry.register_hparams
# # def transformer_librispeech_tpu():
# #   """HParams for training ASR model on Librispeech on TPU."""
# #   return transformer_librispeech_tpu_v2()
# #
# #
# # @registry.register_hparams
# # def transformer_common_voice():
# #   """HParams for training ASR model on Mozilla Common Voice."""
# #   return transformer_librispeech()
# #
# #
# # @registry.register_hparams
# # def transformer_common_voice_tpu():
# #   """HParams for training ASR model on Mozilla Common Voice on TPU."""
# #   hparams = transformer_librispeech_tpu()
# #   hparams.batch_size = 8
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_supervised_attention():
# #   """HParams for supervised attention problems."""
# #   hparams = transformer_base()
# #   # Attention loss type (KL-divergence or MSE).
# #   hparams.add_hparam("expected_attention_loss_type", "kl_divergence")
# #   # Multiplier to the encoder-decoder expected attention loss.
# #   hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_tpu_1b():
# #   """Hparams for machine translation with ~1.1B parameters."""
# #   hparams = transformer_tpu()
# #   hparams.hidden_size = 2048
# #   hparams.filter_size = 8192
# #   hparams.num_hidden_layers = 8
# #   # smaller batch size to avoid OOM
# #   hparams.batch_size = 1024
# #   hparams.activation_dtype = "bfloat16"
# #   hparams.weight_dtype = "bfloat16"
# #   # maximize number of parameters relative to computation by not sharing.
# #   hparams.shared_embedding_and_softmax_weights = False
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_wikitext103_l4k_v0():
# #   """HParams for training languagemodel_wikitext103_l4k."""
# #   hparams = transformer_big()
# #
# #   # Adafactor uses less memory than Adam.
# #   # switch to Adafactor with its recommended learning rate scheme.
# #   hparams.optimizer = "Adafactor"
# #   hparams.learning_rate_schedule = "rsqrt_decay"
# #   hparams.learning_rate_warmup_steps = 10000
# #
# #   hparams.num_heads = 4
# #   hparams.max_length = 4096
# #   hparams.batch_size = 4096
# #   hparams.shared_embedding_and_softmax_weights = False
# #
# #   hparams.num_hidden_layers = 8
# #   hparams.attention_dropout = 0.1
# #   hparams.layer_prepostprocess_dropout = 0.2
# #   hparams.relu_dropout = 0.1
# #   hparams.label_smoothing = 0.0
# #
# #   # Using noise broadcast in the dropout layers saves memory during training.
# #   hparams.attention_dropout_broadcast_dims = "0,1"  # batch, heads
# #   hparams.relu_dropout_broadcast_dims = "1"  # length
# #   hparams.layer_prepostprocess_dropout_broadcast_dims = "1"  # length
# #
# #   # Avoid an expensive concat on TPU.
# #   # >1 shards helps with faster parameter distribution on multi-GPU machines
# #   hparams.symbol_modality_num_shards = 1
# #
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_wikitext103_l4k_memory_v0():
# #   """HParams for training languagemodel_wikitext103_l4k with memory."""
# #   hparams = transformer_wikitext103_l4k_v0()
# #
# #   hparams.split_targets_chunk_length = 64
# #   hparams.split_targets_max_chunks = 64
# #   hparams.add_hparam("memory_type", "transformer_xl")
# #
# #   # The hparams specify batch size *before* chunking, but we want to have a
# #   # consistent 4K batch size *after* chunking to fully utilize the hardware.
# #   target_tokens_per_batch = 4096
# #   hparams.batch_size = int(target_tokens_per_batch * (
# #       hparams.max_length / hparams.split_targets_chunk_length))  # 262144
# #
# #   hparams.pos = None
# #   hparams.self_attention_type = "dot_product_relative"
# #   hparams.max_relative_position = 2 * hparams.split_targets_chunk_length
# #
# #   hparams.add_hparam("unconditional", True)
# #   hparams.add_hparam("recurrent_memory_batch_size", 0)  # 0 = try to guess
# #   # By default, cache one chunk only (like Transformer-XL)
# #   hparams.add_hparam("num_memory_items", hparams.split_targets_chunk_length)
# #
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_wikitext103_l16k_memory_v0():
# #   """HParams for training languagemodel_wikitext103_l16k with memory."""
# #   hparams = transformer_wikitext103_l4k_memory_v0()
# #
# #   hparams.max_length = 16384
# #   hparams.split_targets_chunk_length = 64
# #   hparams.split_targets_max_chunks = int(
# #       hparams.max_length / hparams.split_targets_chunk_length)
# #
# #   # The hparams specify batch size *before* chunking, but we want to have a
# #   # consistent 4K batch size *after* chunking to fully utilize the hardware.
# #   target_tokens_per_batch = 4096
# #   hparams.batch_size = int(target_tokens_per_batch * (
# #       hparams.max_length / hparams.split_targets_chunk_length))
# #
# #   hparams.max_relative_position = 2 * hparams.split_targets_chunk_length
# #
# #   return hparams
# #
# #
# # @registry.register_hparams
# # def transformer_cifar10_memory_v0():
# #   """HParams for training image_cifar10_plain_gen_flat_rev with memory."""
# #   hparams = transformer_wikitext103_l4k_memory_v0()
# #
# #   hparams.num_hidden_layers = 6
# #
# #   hparams.max_length = 32 * 32 * 3
# #   hparams.split_targets_chunk_length = 64 * 3
# #   hparams.split_targets_max_chunks = int(
# #       hparams.max_length / hparams.split_targets_chunk_length)
# #   hparams.num_memory_items = 128 * 3
# #
# #   # Since this is an image problem, batch size refers to examples (not tokens)
# #   target_images_per_batch = 4
# #   hparams.batch_size = int(target_images_per_batch * (
# #       hparams.max_length / hparams.split_targets_chunk_length))
# #
# #   # The recurrent memory needs to know the actual batch size (in sequences)
# #   hparams.recurrent_memory_batch_size = hparams.batch_size
# #
# #   hparams.max_relative_position = (
# #       hparams.num_memory_items + hparams.split_targets_chunk_length)
# #
# #   return hparams
