import tensorflow as tf
from TFNetworkLayer import LayerBase, _ConcatInputLayer, Loss, get_concat_sources_data_template
from TFNetworkRecLayer import RecLayer
from TFUtil import Data, sparse_labels_with_seq_lens
from Util import softmax


class NeuralTransducerLayer(_ConcatInputLayer):
    """
    Creates a neural transducer based on the paper "A Neural Transducer": https://arxiv.org/abs/1511.04868.
    NOTE: Requires that the loss be neural_transducer_loss.
    NOTE: When training with BiLSTM as input, set an appropriate gradient clipping parameter.
    """
    layer_class = "neural_transducer"

    def __init__(self, transducer_hidden_units, n_out, transducer_max_width, input_block_size,
                 embedding_size, e_symbol_index, use_prev_state_as_start=False, **kwargs):
        """
        Initialize the Neural Transducer.
        :param int transducer_hidden_units: Amount of units the transducer should have.
        :param int n_out: The size of the output layer, i.e. the size of the vocabulary including <E> symbol.
        :param int transducer_max_width: The max amount of outputs in one NT block (including the final <E> symbol)
        :param int input_block_size: Amount of inputs to use for each NT block.
        :param int embedding_size: Embedding dimension size.
        :param int e_symbol_index: Index of e symbol that is used in the NT block. 0 <= e_symbol_index < num_outputs
        :param bool use_prev_state_as_start: Whether to use the last state of the previous recurrent layer as the ]
        initial state of the transducer. NOTE: For this to work, you have to watch out for:
        previous_layer.hidden_units = previous_layer.n_out = transducer.transducer_hidden_units
        """

        super(NeuralTransducerLayer, self).__init__(**kwargs)

        # TODO: Build optimized version

        # Get embedding
        from TFUtil import get_initializer
        initializer = get_initializer('glorot_uniform',
                                      seed=self.network.random.randint(2 ** 31),
                                      eval_local_ns={"layer": self})
        embeddings = self.add_param(tf.get_variable(shape=[n_out, embedding_size], dtype=tf.float32,
                                                    initializer=initializer, name='nt_embedding'),
                                    trainable=True, saveable=True)

        # Ensure encoder is time major
        encoder_outputs = self.input_data.get_placeholder_as_time_major()

        # Pad encoder outputs with zeros so that it its cleanly divisible by the input_block_size
        batch_size = tf.shape(encoder_outputs)[1]
        time_length_to_append = input_block_size - tf.mod(tf.shape(encoder_outputs)[0], input_block_size)
        padding_tensor = tf.zeros([time_length_to_append, batch_size, tf.shape(encoder_outputs)[2]],
                                  dtype=tf.float32)
        encoder_outputs = tf.concat([encoder_outputs, padding_tensor], axis=0)
        # Do assertions
        assert 0 <= e_symbol_index < n_out, 'NT: E symbol outside possible outputs!'

        # Get prev state as start state
        last_hidden = None
        if use_prev_state_as_start is True and isinstance(self.sources[0], RecLayer) is True:
            # TODO: add better checking whether the settings are correct
            last_hidden_c = self.sources[0].get_last_hidden_state('*')  # Get last c after all blocks
            last_hidden_h = encoder_outputs[input_block_size - 1]  # Get last hidden after the first block

            # Padding so that last hidden_c & _h are the same (this is needed for when using BiLSTM)
            c_shape = tf.shape(last_hidden_c)
            h_shape = tf.shape(last_hidden_h)
            padding = tf.zeros([c_shape[0], h_shape[1] - c_shape[1]])
            last_hidden_c = tf.concat([last_hidden_c, padding], axis=1)

            last_hidden = tf.stack([last_hidden_c, last_hidden_h], axis=0)

        # Note down data
        self.transducer_hidden_units = transducer_hidden_units
        self.num_outputs = n_out
        self.transducer_max_width = transducer_max_width
        self.input_block_size = input_block_size
        self.e_symbol_index = e_symbol_index

        # self.output.placeholder is of shape [transducer_max_width * amount_of_blocks, batch_size, n_out]
        self.output.placeholder = self.build_full_transducer(transducer_hidden_units=transducer_hidden_units,
                                                             embeddings=embeddings,
                                                             num_outputs=n_out,
                                                             input_block_size=input_block_size,
                                                             transducer_max_width=transducer_max_width,
                                                             encoder_outputs=encoder_outputs,
                                                             trans_hidden_init=last_hidden)

        # Set correct logit lengths
        output_size = self.round_vector_to_closest_input_block(vector=self.input_data.size_placeholder[0],
                                                               input_block_size=input_block_size,
                                                               transducer_max_width=transducer_max_width)

        # Set shaping info
        self.output.size_placeholder = {
          0: output_size
        }
        self.output.time_dim_axis = 0
        self.output.batch_dim_axis = 1

        # Add all trainable params
        with self.var_creation_scope() as scope:
            self._add_all_trainable_params(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope.name))

    def build_full_transducer(self, transducer_hidden_units, embeddings, num_outputs, input_block_size,
                              transducer_max_width, encoder_outputs, trans_hidden_init):
        """
        Builds the complete transducer.
        :param int transducer_hidden_units:  Amount of units the transducer should have.
        :param tf.Variable embeddings: Variable with the reference to the embeddings.
        :param int num_outputs: The size of the output layer, i.e. the size of the vocabulary including <E> symbol.
        :param int input_block_size: Amount of inputs to use for each NT block.
        :param int transducer_max_width: The max amount of outputs in one NT block (including the final <E> symbol)
        :param tf.tensor encoder_outputs: The outputs of the encode in shape of [max_time, batch_size, encoder_hidden]
        :param tf.tensor trans_hidden_init: The init state of the transducer. Needs to be of shape
        [2, batch_size, transducer_hidden_units]. The trans_hidden_init[0] is the c vector of the lstm,
        trans_hidden_init[1] the hidden vector.
        :return: Returns a reference to the tf.tensor containing the logits.
        :rtype: tf.tensor
        """

        with self.var_creation_scope():
            # Get meta variables
            batch_size = tf.shape(encoder_outputs)[1]
            if trans_hidden_init is None:
                trans_hidden_init = tf.zeros([2, batch_size, transducer_hidden_units], dtype=tf.float32)

            # Do some more post processing
            max_blocks = tf.to_int32(tf.shape(encoder_outputs)[0]/input_block_size)
            transducer_list_outputs = tf.ones([max_blocks, batch_size], dtype=tf.int32) * transducer_max_width
            inference_mode = 1.0
            teacher_forcing_targets = tf.ones([transducer_max_width * max_blocks, batch_size], dtype=tf.int32)

            # Process teacher forcing targets
            teacher_forcing_targets_emb = tf.nn.embedding_lookup(embeddings, teacher_forcing_targets)

            # Outputs
            outputs_ta = tf.TensorArray(dtype=tf.float32, size=max_blocks, infer_shape=False)
            init_state = (0, outputs_ta, trans_hidden_init, 0)

            # Init the transducer cell
            from TFUtil import get_initializer
            transducer_cell_initializer = get_initializer('glorot_uniform',
                                                          seed=self.network.random.randint(2 ** 31),
                                                          eval_local_ns={"layer": self})
            transducer_cell = tf.contrib.rnn.LSTMCell(transducer_hidden_units, initializer=transducer_cell_initializer)

            def cond(current_block, outputs_int, trans_hidden, total_output):
                return current_block < max_blocks

            def body(current_block, outputs_int, trans_hidden, total_output):

                # --------------------- TRANSDUCER --------------------------------------------------------------------
                # Each transducer block runs for the max transducer outputs in its respective block

                encoder_raw_outputs = encoder_outputs[input_block_size * current_block:
                                                      input_block_size * (current_block + 1)]

                encoder_raw_outputs = tf.where(tf.is_nan(encoder_raw_outputs), tf.zeros_like(encoder_raw_outputs),
                                               encoder_raw_outputs)

                trans_hidden = tf.where(tf.is_nan(trans_hidden), tf.zeros_like(trans_hidden), trans_hidden)

                # Save/load the state as one tensor, use top encoder layer state as init if this is the first block
                trans_hidden_state = trans_hidden
                transducer_amount_outputs = transducer_list_outputs[current_block]
                transducer_max_output = tf.reduce_max(transducer_amount_outputs)

                # Model building
                helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
                    inputs=teacher_forcing_targets_emb[total_output:total_output + transducer_max_output],  # Get the current target inputs
                    sequence_length=transducer_amount_outputs,
                    embedding=embeddings,
                    sampling_probability=inference_mode,
                    time_major=True
                )

                attention_states = tf.transpose(encoder_raw_outputs,
                                                [1, 0, 2])  # attention_states: [batch_size, max_time, num_enc_units]
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    transducer_hidden_units, attention_states)

                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
                    transducer_cell,
                    attention_mechanism,
                    attention_layer_size=transducer_hidden_units)

                from tensorflow.python.layers import core as layers_core
                projection_layer = layers_core.Dense(num_outputs, use_bias=False)

                # Build previous state
                trans_hidden_c, trans_hidden_h = tf.split(trans_hidden_state, num_or_size_splits=2, axis=0)
                trans_hidden_c = tf.reshape(trans_hidden_c, shape=[-1, transducer_hidden_units])
                trans_hidden_h = tf.reshape(trans_hidden_h, shape=[-1, transducer_hidden_units])
                from tensorflow.contrib.rnn import LSTMStateTuple
                trans_hidden_state_t = LSTMStateTuple(trans_hidden_c, trans_hidden_h)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper,
                    decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=trans_hidden_state_t),
                    output_layer=projection_layer)
                outputs, transducer_hidden_state_new, _ = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                            output_time_major=True,
                                                                                            maximum_iterations=transducer_max_output)
                logits = outputs.rnn_output  # logits of shape [max_time,batch_size,vocab_size]

                # Modify output of transducer_hidden_state_new so that it can be fed back in again without problems.

                transducer_hidden_state_new = tf.concat(
                    [transducer_hidden_state_new[0].c, transducer_hidden_state_new[0].h],
                    axis=0)
                transducer_hidden_state_new = tf.reshape(transducer_hidden_state_new,
                                                         shape=[2, -1, transducer_hidden_units])

                # Note the outputs
                outputs_int = outputs_int.write(current_block, logits)

                return current_block + 1, outputs_int, \
                    transducer_hidden_state_new, total_output + transducer_max_output

            _, outputs_final, _, _ = tf.while_loop(cond, body, init_state, parallel_iterations=1)

            # Process outputs
            with tf.device('/cpu:0'):
              logits = outputs_final.concat()  # And now its [max_output_time, batch_size, num_outputs]

            # For loading the model later on
            logits = tf.identity(logits, name='logits')

        return logits

    def _add_all_trainable_params(self, tf_vars):
        for var in tf_vars:
            self.add_param(param=var, trainable=True, saveable=True)

    def round_vector_to_closest_input_block(self, vector, input_block_size, transducer_max_width):
        """
        Rounds up the provided vector so that every entry is a multiple of input_block_size.
        :param tf.tensor vector: A vector.
        :param int input_block_size: Input block size as specified in the __init__ function.
        :return: tf.tensor A vector the same shape as 'vector'.
        """
        vector = tf.cast(tf.ceil(tf.cast(vector, tf.float32) / input_block_size), tf.float32) * tf.cast(transducer_max_width, tf.float32)
        vector = tf.cast(vector, tf.int32)

        return vector


    @classmethod
    def get_out_data_from_opts(cls, n_out, **kwargs):
        data = get_concat_sources_data_template(kwargs["sources"], name="%s_output" % kwargs["name"])
        data = data.copy_as_time_major()  # type: Data
        data.shape = (None, n_out)
        data.time_dim_axis = 0
        data.batch_dim_axis = 1
        data.dim = n_out
        return data


class NeuralTransducerLoss(Loss):
    """
    The loss function that should be used with the NeuralTransducer layer. This loss function has the built in
    alignment algorithm from the original paper.
    """
    class_name = "neural_transducer"
    recurrent = True

    class Alignment(object):
        """
        Class to manage the alignment generation in the NT.
        """

        def __init__(self, transducer_hidden_units, E_SYMBOL):
            """
            Alignment initiation.
            :param int transducer_hidden_units: Amount of hidden units that the transducer should have.
            :param int E_SYMBOL: The index of the <e> symbol.
            """
            import numpy
            self.alignment_position = (0, 1)  # first entry is position in target (y~), second is the block index
            self.log_prob = 0  # The sum log prob of this alignment over the target indices
            self.alignment_locations = []  # At which indices in the target output we need to insert <e>
            self.last_state_transducer = numpy.zeros(
                shape=(2, 1, transducer_hidden_units))  # Transducer state
            self.E_SYMBOL = E_SYMBOL  # Index of

        def __compute_sum_probabilities(self, transducer_outputs, targets, transducer_amount_outputs):
            """
            # TODO move this function outside the Alignment class scope
            Computes the sum log probabilities of the outputs based on the targets.
            :param np.ndarray[int] transducer_outputs: Softmaxed transducer outputs of one block.
            Size: [transducer_amount_outputs, 1, num_outputs]
            :param [int] targets: List of targets.
            :param int transducer_amount_outputs: The width of this transducer block.
            :return: The summed log prob for this block.
            :rtype: float
            """
            import numpy

            def get_prob_at_timestep(timestep):
                if timestep + start_index < len(targets):
                    # For normal operations
                    if transducer_outputs[timestep][0][targets[start_index + timestep]] <= 0:
                        return -10000000.0 + numpy.random.uniform(-100, -500)  # Some large negative number
                    else:
                        return numpy.log(transducer_outputs[timestep][0][targets[start_index + timestep]])
                else:
                    # For last timestep, so the <e> symbol
                    if transducer_outputs[timestep][0][self.E_SYMBOL] <= 0:
                        return -10000000.0 + numpy.random.uniform(-100, -500)  # Some large negative number
                    else:
                        return numpy.log(transducer_outputs[timestep][0][self.E_SYMBOL])

            # print transducer_outputs
            start_index = self.alignment_position[
                              0] - transducer_amount_outputs  # The current position of this alignment
            prob = 0
            for i in range(0,
                           transducer_amount_outputs + 1):  # Do not include e symbol in calculation, +1 due to last symbol
                prob += get_prob_at_timestep(i)
            return prob

        def insert_alignment(self, index, block_index, transducer_outputs, targets, transducer_amount_outputs,
                             new_transducer_state):
            """
            Inserts alignment properties for a new block.
            :param int index: The index of of y~ corresponding to the last target index.
            :param int block_index: The new block index.
            :param np.ndarray transducer_outputs: The computed transducer outputs. Shape
            [transducer_amount_outputs, 1, n_out]
            :param np.ndarray targets: The complete target array, should be of shape [total_target_length].
            :param int transducer_amount_outputs: The amount of outputs that the transducer created in this block.
            :param np.ndarray new_transducer_state: The new transducer state of shape [2, 1, transducer_hidden_units]
            """
            self.alignment_locations.append(index)
            self.alignment_position = (index, block_index)
            self.log_prob += self.__compute_sum_probabilities(transducer_outputs, targets, transducer_amount_outputs)
            self.last_state_transducer = new_transducer_state

    def __init__(self, debug=False, max_variance=999999.9, **kwargs):
        """
        Initialize the Neural Transducer loss.
        :param bool debug: Whether to output debug info such as alignments, argmax, variance etc...
        :param float max_variance: If a time step (in CE) has a too high variance in within the batch, then the gradient
        for that time step will be ignored. Set this value lower if you have outliers that disrupt training.
        """
        super(NeuralTransducerLoss, self).__init__(**kwargs)

        self.transducer_hidden_units = 0
        self.num_outputs = 0
        self.transducer_max_width = 0
        self.input_block_size = 0
        self.e_symbol_index = 0
        self.debug = debug
        self.reduce_func = tf.reduce_sum
        self.max_variance = max_variance

    def init(self, **kwargs):
        super(NeuralTransducerLoss, self).init(**kwargs)
        # Get setup vars from sources
        base_class = None
        for c in self.base_network.layers:
            if isinstance(self.base_network.layers[c], NeuralTransducerLayer):
                base_class = self.base_network.layers[c]

        assert base_class is not None, "Neural Transducer layer not found!"
        self.transducer_hidden_units = base_class.transducer_hidden_units
        self.num_outputs = base_class.num_outputs
        self.transducer_max_width = base_class.transducer_max_width
        self.input_block_size = base_class.input_block_size
        self.e_symbol_index = base_class.e_symbol_index

    def get_value(self):
        logits = self.output.copy_as_time_major().placeholder
        logits_lengths = self.output.size_placeholder[0]
        targets = self.target.copy_as_time_major().placeholder
        targets_lengths = self.target.size_placeholder[0]

        # Get alignment info into our targets
        new_targets, mask = tf.py_func(func=self.get_alignment_from_logits_manager,
                                       inp=[logits, targets, logits_lengths, targets_lengths],
                                       Tout=(tf.int64, tf.bool), stateful=False)

        # Get CE
        stepwise_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=new_targets, logits=logits)

        # Debugging
        if self.debug is True:
            stepwise_cross_entropy = tf.Print(stepwise_cross_entropy, [targets[:, 0]], message='Targets: ',
                                              summarize=1000)
            stepwise_cross_entropy = tf.Print(stepwise_cross_entropy, [new_targets[:, 0]],
                                              message='Targets with alignment: ', summarize=1000)
            stepwise_cross_entropy = tf.Print(stepwise_cross_entropy, [tf.argmax(logits, axis=2)[:, 0]],
                                              message='Argmax: ', summarize=1000)

        # Apply masking step AFTER cross entropy:
        zeros = tf.zeros_like(stepwise_cross_entropy)
        stepwise_cross_entropy = tf.where(mask, stepwise_cross_entropy, zeros)

        # Check for outliers and set their gradient to 0
        loss_time = tf.reduce_sum(stepwise_cross_entropy, axis=1)
        mean, variance = tf.nn.moments(stepwise_cross_entropy, axes=[1])
        loss_mask = tf.to_float(variance > self.max_variance)
        stepwise_cross_entropy = tf.stop_gradient(tf.multiply(loss_mask, loss_time)) + \
                                  tf.multiply(tf.to_float(tf.logical_not(tf.cast(loss_mask, tf.bool))), loss_time)

        if self.debug is True:
            stepwise_cross_entropy = tf.cond(tf.reduce_sum(loss_mask) >= 1,
                                             lambda: tf.Print(stepwise_cross_entropy, [variance, loss_mask],
                                                              message='High Variance: ', summarize=500),
                                             lambda: stepwise_cross_entropy)

        # TODO: add forwarding layer

        # Get full loss
        norm = tf.to_float(tf.reduce_sum(targets_lengths)) / tf.reduce_sum(tf.to_float(mask))
        loss = tf.reduce_sum(stepwise_cross_entropy) * norm

        return loss

    def get_alignment_from_logits(self, logits, targets, amount_of_blocks, transducer_max_width):
        """
        Finds the alignment of the target sequence to the actual output.
        :param logits: Logits from transducer, of size [transducer_max_width * amount_of_blocks, 1, vocab_size]
        :param targets: The target sequence of shape [time] where each entry is an index.
        :param amount_of_blocks: Amount of blocks in Neural Transducer.
        :param transducer_max_width: The max width of one transducer block.
        :return: Returns a list of indices where <e>'s need to be inserted into the target sequence, shape: [max_time, 1]
        (see paper) and a boolean mask for use with a loss function of shape [max_time, 1].
        """
        import numpy
        import copy
        # Split logits into list of arrays with each array being one block
        # of shape [transducer_max_width, 1, vocab_size]
        logits = numpy.reshape(logits, newshape=[logits.shape[0], 1, logits.shape[1]])

        split_logits = numpy.split(logits, amount_of_blocks)

        # print 'Raw logits: ' + str(softmax(split_logits[0][0:transducer_max_width], axis=2))

        def run_new_block(previous_alignments, block_index, transducer_max_width, targets,
                          total_blocks):
            """
            Runs one block of the alignment process.
            :param previous_alignments: List of alignment objects from previous block step.
            :param block_index: The index of the current new block.
            :param transducer_max_width: The max width of the transducer block.
            :param targets: The full target array of shape [time]
            :param total_blocks: The total amount of blocks.
            :return: new_alignments as list of Alignment objects
            """

            def run_transducer(current_block, transducer_width):
                # apply softmax on the correct outputs
                transducer_out = softmax(split_logits[current_block][0:transducer_width], axis=2)
                return transducer_out

            # Look into every existing alignment
            new_alignments = []
            for i in range(len(previous_alignments)):
                alignment = previous_alignments[i]

                # Expand the alignment for each transducer width, only look at valid options
                targets_length = len(targets)

                min_index = alignment.alignment_position[0] + transducer_max_width + \
                            max(-transducer_max_width,
                                targets_length - ((total_blocks - block_index + 1) * transducer_max_width
                                                  + alignment.alignment_position[0]))
                max_index = alignment.alignment_position[0] + transducer_max_width + min(0, targets_length - (
                    alignment.alignment_position[0] + transducer_max_width))

                # new_alignment_index's value is equal to the index of y~ for that computation
                for new_alignment_index in range(min_index, max_index + 1):  # 1 so that the max_index is also used
                    # Create new alignment
                    new_alignment = copy.deepcopy(alignment)
                    new_alignment_width = new_alignment_index - new_alignment.alignment_position[0]
                    trans_out = run_transducer(transducer_width=new_alignment_width + 1, current_block=block_index - 1)

                    new_alignment.insert_alignment(new_alignment_index, block_index, trans_out, targets,
                                                   new_alignment_width, None)
                    new_alignments.append(new_alignment)

            # Delete all overlapping alignments, keeping the highest log prob
            for a in reversed(new_alignments):
                for o in new_alignments:
                    if o is not a and a.alignment_position == o.alignment_position and o.log_prob > a.log_prob:
                        if a in new_alignments:
                            new_alignments.remove(a)

            assert len(new_alignments) > 0, 'Error in amount of alignments! %s' % str(targets)

            return new_alignments

        # Manage variables
        current_block_index = 1
        current_alignments = [self.Alignment(transducer_hidden_units=self.transducer_hidden_units,
                                             E_SYMBOL=self.e_symbol_index)]

        # Do assertions to check whether everything was correctly set up.
        assert (transducer_max_width - 1) * amount_of_blocks >= len(
            targets), 'transducer_max_width to small for targets'

        for block in range(current_block_index, amount_of_blocks + 1):
            # Run all blocks
            current_alignments = run_new_block(previous_alignments=current_alignments,
                                               block_index=block,
                                               transducer_max_width=transducer_max_width - 1,  # -1 due to offset for e
                                               targets=targets, total_blocks=amount_of_blocks)
            # for alignment in current_alignments:

        # Select first alignment if we have multiple with the same log prob (happens with ~1% probability in training)
        if self.debug is True:
            print('Alignment: ' + str(current_alignments[0].alignment_locations) + ' for targets: ' + str(targets))

        def modify_targets(targets, alignment):
            # Calc lengths for each transducer block
            lengths_temp = []
            alignment.insert(0, 0)  # This is so that the length calculation is done correctly
            for i in range(1, len(alignment)):
                lengths_temp.append(alignment[i] - alignment[i - 1] + 1)
            del alignment[0]  # Remove alignment index that we added
            lengths = lengths_temp

            # Modify targets so that it has the appropriate alignment
            offset = 0
            for e in alignment:
                targets.insert(e + offset, self.e_symbol_index)
                offset += 1

            # Modify so that all targets have same lengths in each transducer block using 0 (will be masked away)
            offset = 0
            for i in range(len(alignment)):
                for app in range(transducer_max_width - lengths[i]):
                    targets.insert(offset + lengths[i], 0)
                offset += transducer_max_width

            # Process targets back to time major
            targets = numpy.asarray([targets])
            targets = numpy.transpose(targets, axes=[1, 0])

            return targets, lengths

        m_targets, lengths = modify_targets(targets.tolist(), current_alignments[0].alignment_locations)
        # m_targets now of shape: [max_time, 1 (batch_size)] = [transducer_max_width * number_of_blocks, 1]

        # Create boolean mask for TF so that unnecessary logits are not used for the loss function
        # Of shape [max_time, batch_size], True where gradient data is kept, False where not

        def create_mask(lengths):
            mask = numpy.full(m_targets.shape, False)
            for i in range(amount_of_blocks):
                for j in range(lengths[i]):
                    mask[i*transducer_max_width:i*transducer_max_width + j + 1, 0] = True
            return mask

        mask = create_mask(lengths)

        return m_targets, mask

    def get_alignment_from_logits_manager(self, logits, targets, logit_lengths, targets_lengths):
        """
        Get the modified targets & mask.
        :param logits: Logits of shape [max_time, batch_size, vocab_size]
        :param targets: Targets of shape [max_time, batch_size]. Each entry denotes the index of the correct target.
        :return: modified targets of shape [max_time, batch_size, vocab_size]
        & mask of shape [max_time, batch_size]
        """
        import numpy
        logits = numpy.copy(logits)
        targets = numpy.copy(targets)

        # print('Targets: ' + str(targets), file=log.v1)

        m_targets = []
        masks = []

        # amount_of_blocks = int(logits.shape[0]/self.transducer_max_width)

        # Go over every sequence in batch
        for batch_index in range(logits.shape[1]):
            # Slice correct logits & targets
            logit_length = logit_lengths[batch_index]
            target_length = targets_lengths[batch_index]
            amount_of_blocks = int(logit_length/self.transducer_max_width)

            temp_target, temp_mask = self.get_alignment_from_logits(logits=logits[0:logit_length, batch_index, :],
                                                                    targets=targets[0:target_length, batch_index],
                                                                    amount_of_blocks=amount_of_blocks,
                                                                    transducer_max_width=self.transducer_max_width)
            # Pad afterwards each target (based on targets_lengths) & mask (based on logit_lengths)
            temp_target = numpy.append(temp_target, numpy.zeros(shape=(logits.shape[0] - temp_target.shape[0], 1), dtype=int), axis=0)
            temp_mask = numpy.append(temp_mask, numpy.zeros(shape=(logits.shape[0] - logit_length, 1), dtype=bool), axis=0)

            m_targets.append(temp_target)
            masks.append(temp_mask)

        # Concatenate the targets & masks on the time axis; due to padding m_targets are all the same
        m_targets = numpy.concatenate(m_targets, axis=1)
        masks = numpy.concatenate(masks, axis=1)

        return m_targets, masks

    @classmethod
    def get_auto_output_layer_dim(cls, target_dim):
        return target_dim + 1  # one added for <E>

    def get_error(self):
        with tf.name_scope("loss_frame_error"):
            logits = self.output.copy_as_time_major().placeholder
            logits_lengths = self.output.size_placeholder[0]
            targets = self.target.copy_as_time_major().placeholder
            targets_lengths = self.target.size_placeholder[0]

            # Get alignment info into our targets
            new_targets, mask = tf.py_func(func=self.get_alignment_from_logits_manager,
                                           inp=[logits, targets, logits_lengths, targets_lengths],
                                           Tout=(tf.int64, tf.bool), stateful=False)

            output_label = tf.cast(tf.argmax(logits, axis=2), tf.int64)
            zeros = tf.zeros_like(output_label)

            # Calculate edit distance
            # First modify outputs so that only those outputs in the mask are considered
            mod_logits = tf.where(mask, output_label, zeros)

            # Get find seq lens (due to having blank spaces in the modified targets we need to use this method to get
            # the correct seq lens)
            seq_lens = tf.argmax(tf.cumsum(tf.to_int32(mask), axis=0), axis=0)
            seq_lens = tf.reshape(seq_lens, shape=[tf.shape(seq_lens)[0]])

            logits_sparse = sparse_labels_with_seq_lens(tf.transpose(mod_logits), seq_lens=seq_lens)
            targets_sparse = sparse_labels_with_seq_lens(tf.transpose(new_targets), seq_lens=seq_lens)
            
            e = tf.edit_distance(logits_sparse[0], targets_sparse[0], normalize=False)
            total = tf.reduce_sum(e)

            norm = tf.to_float(tf.reduce_sum(targets_lengths)) / tf.reduce_sum(tf.to_float(mask))
            total = total * norm

            return total
