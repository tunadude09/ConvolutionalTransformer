
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




#  EXTRA CODE IS HERE, EVERYTHING ELSE IS THE SAME





    # Drop out attention links for each head.
    weights = common_layers.dropout_with_broadcast_dims(
        weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)

    if common_layers.should_generate_summaries() and make_image_summary:
      attention_image_summary(weights, image_shapes)
    return tf.matmul(weights, v)
