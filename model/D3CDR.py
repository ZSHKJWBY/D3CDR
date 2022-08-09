import sys
sys.path.append('../')
import params
import tensorflow as tf
import os
from progressbar import *
from tensorflow.contrib import layers
from utils import *
import pickle

test = False
test_model = 'data/amazon/Book_CD/model_CoNet/-1881'

fusion_size = 3
model_save_name = "/D3CDR/"


class D3CDR(object):
    def __init__(self, num_users, num_items_1, num_items_2, dim, num_cross_layers, num_whole_layer,
                 layer_dim, beta, lr, activation, fusion_size=2):
        self.num_cross_layers = num_cross_layers
        self.num_whole_layers = num_whole_layer
        self.dim = dim
        self.layers = layer_dim

        self.n_users = num_users
        self.n_items_1 = num_items_1
        self.n_items_2 = num_items_2
        self.lr = lr
        self.activation = activation
        self.beta = beta
        self.init_std = 0.01

        self.users_1 = tf.placeholder(tf.int32, [None])
        self.users_2 = tf.placeholder(tf.int32, [None])
        # domain 1 corresponding positive samples and negative samples
        self.pos_items_1 = tf.placeholder(tf.int32, [None])
        self.neg_items_1 = tf.placeholder(tf.int32, [None])
        # domain 2 corresponding positive samples and negative samples
        self.pos_items_2 = tf.placeholder(tf.int32, [None])
        self.neg_items_2 = tf.placeholder(tf.int32, [None])

        self.keep_prob = tf.placeholder(dtype=tf.float32)

        self.initializer = tf.contrib.layers.xavier_initializer()
        self.fusion_size = fusion_size
        # initialized params matrix for users and items
        self.user_embeddings = tf.Variable(tf.random_normal([self.n_users, self.dim], stddev=self.init_std))
        self.item_embeddings_1 = tf.Variable(tf.random_normal([self.n_items_1, self.dim], stddev=self.init_std))
        self.item_embeddings_2 = tf.Variable(tf.random_normal([self.n_items_2, self.dim], stddev=self.init_std))

        self.SWeight1 = defaultdict(object)
        self.SWeight2 = defaultdict(object)
        self.attentionWeight1 = defaultdict(object)
        self.attentionWeight2 = defaultdict(object)
        self.attentionWeight1_new = defaultdict(object)
        self.attentionWeight2_new = defaultdict(object)
        self.gateWeight_domain1 = defaultdict(object)
        self.gateWeight_domain2 = defaultdict(object)
        self.gateBias_domain1 = defaultdict(object)
        self.gateBias_domain2 = defaultdict(object)
        
        for h in range(1, self.num_cross_layers + 1):
            # [64, 32*4] [32, 16*4]
            self.SWeight1[h] = (tf.Variable(
                tf.random_normal([self.layers[h - 1], (self.layers[h] * self.fusion_size)], stddev=self.init_std)))
            # [64, 32*4] [32, 16*4]
            self.SWeight2[h] = (tf.Variable(
                tf.random_normal([self.layers[h - 1], (self.layers[h] * self.fusion_size)], stddev=self.init_std)))
            # [32*4, 4] [16*4, 4]
            self.attentionWeight1[h] = (tf.Variable(
                tf.random_normal([self.layers[h] * self.fusion_size, self.fusion_size],
                                 stddev=self.init_std)))
            # [32*4, 4] [16*4, 4]
            self.attentionWeight2[h] = (tf.Variable(
                tf.random_normal([self.layers[h] * self.fusion_size, self.fusion_size],
                                 stddev=self.init_std)))

            self.attentionWeight1_new[h] = (tf.Variable(
                tf.random_normal([self.layers[h-1]+self.layers[h] * self.fusion_size, self.fusion_size],
                                 stddev=self.init_std)))
            # [32*4, 4] [16*4, 4]
            self.attentionWeight2_new[h] = (tf.Variable(
                tf.random_normal([self.layers[h-1]+self.layers[h] * self.fusion_size, self.fusion_size],
                                 stddev=self.init_std)))

            # [32, 32] [16, 16]
            self.gateWeight_domain1[h] = (tf.Variable(tf.random_normal([self.layers[h], self.layers[h]],
                                                                       stddev=self.init_std)))
            self.gateWeight_domain2[h] = (tf.Variable(tf.random_normal([self.layers[h], self.layers[h]],
                                                                       stddev=self.init_std)))

            self.gateBias_domain1[h] = (tf.Variable(tf.random_normal([self.layers[h]], stddev=self.init_std)))
            self.gateBias_domain2[h] = (tf.Variable(tf.random_normal([self.layers[h]], stddev=self.init_std)))

        with tf.name_scope("domain_1"):
            self.weights_domain1 = defaultdict(object)
            self.bias_domain1 = defaultdict(object)
            for layer_idx in range(1, self.num_whole_layers):
                # [64,32] [32,16] [16,8]
                self.weights_domain1[layer_idx] = (tf.Variable(tf.random_normal([self.layers[layer_idx - 1],
                                                                                 self.layers[layer_idx]],
                                                                                stddev=self.init_std)))
                # [32,] [16,] [8,]
                self.bias_domain1[layer_idx] = (tf.Variable(tf.random_normal([self.layers[layer_idx]], stddev=0.01)))
            # [8,1]
            self.h1 = tf.Variable(tf.random_normal([self.layers[-1], 1], stddev=self.init_std))
            # [1,]
            self.b1 = tf.Variable(tf.random_normal([1], stddev=self.init_std))
        with tf.name_scope("domain_2"):
            self.weights_domain2 = defaultdict(object)
            self.bias_domain2 = defaultdict(object)
            for layer_idx in range(1, self.num_whole_layers):
                # [64,32] [32,16] [16,8]
                self.weights_domain2[layer_idx] = (tf.Variable(tf.random_normal([self.layers[layer_idx - 1],
                                                                                 self.layers[layer_idx]],
                                                                                stddev=self.init_std)))
                # [32,] [16,] [8,]
                self.bias_domain2[layer_idx] = (tf.Variable(tf.random_normal([self.layers[layer_idx]], stddev=0.01)))
            # [8,1]
            self.h2 = tf.Variable(tf.random_normal([self.layers[-1], 1], stddev=self.init_std))
            # [1,]
            self.b2 = tf.Variable(tf.random_normal([1], stddev=self.init_std))

        batch_u_embeddings_1 = tf.nn.embedding_lookup(self.user_embeddings, self.users_1)
        batch_u_embeddings_2 = tf.nn.embedding_lookup(self.user_embeddings, self.users_2)
        batch_pos_i_embeddings_1 = tf.nn.embedding_lookup(self.item_embeddings_1, self.pos_items_1)
        batch_neg_i_embeddings_1 = tf.nn.embedding_lookup(self.item_embeddings_1, self.neg_items_1)

        batch_pos_i_embeddings_2 = tf.nn.embedding_lookup(self.item_embeddings_2, self.pos_items_2)
        batch_neg_i_embeddings_2 = tf.nn.embedding_lookup(self.item_embeddings_2, self.neg_items_2)

        # ui_pos/neg: (1024, 64)
        ui_pos_1 = tf.concat([batch_u_embeddings_1, batch_pos_i_embeddings_1], axis=1)
        ui_neg_1 = tf.concat([batch_u_embeddings_1, batch_neg_i_embeddings_1], axis=1)
        ui_pos_2 = tf.concat([batch_u_embeddings_2, batch_pos_i_embeddings_2], axis=1)
        ui_neg_2 = tf.concat([batch_u_embeddings_2, batch_neg_i_embeddings_2], axis=1)

        self.layer_h_pos_1 = defaultdict(object)
        layer_h_pos_1 = tf.reshape(ui_pos_1, [-1, self.dim * 2])  # (1024, 64)
        self.layer_h_pos_1[0] = layer_h_pos_1

        self.layer_h_pos_2 = defaultdict(object)
        layer_h_pos_2 = tf.reshape(ui_pos_2, [-1, self.dim * 2])
        self.layer_h_pos_2[0] = layer_h_pos_2

        self.layer_h_neg_1 = defaultdict(object)
        layer_h_neg_1 = tf.reshape(ui_neg_1, [-1, self.dim * 2])
        self.layer_h_neg_1[0] = layer_h_neg_1

        self.layer_h_neg_2 = defaultdict(object)
        layer_h_neg_2 = tf.reshape(ui_neg_2, [-1, self.dim * 2])
        self.layer_h_neg_2[0] = layer_h_neg_2

        regu_inner_pos1 = 0
        regu_inner_neg1 = 0
        regu_inner_pos2 = 0
        regu_inner_neg2 = 0

        for h in range(1, self.num_whole_layers):
            layer_h_pos_1 = tf.add(tf.layers.dropout(tf.matmul(self.layer_h_pos_1[h - 1], self.weights_domain1[h]),
                                                     rate=self.keep_prob), self.bias_domain1[h])
            layer_h_neg_1 = tf.add(tf.layers.dropout(tf.matmul(self.layer_h_neg_1[h - 1], self.weights_domain1[h]),
                                                     rate=self.keep_prob), self.bias_domain1[h])

            layer_h_pos_2 = tf.add(tf.layers.dropout(tf.matmul(self.layer_h_pos_2[h - 1], self.weights_domain2[h]),
                                   rate=self.keep_prob), self.bias_domain2[h])
            layer_h_neg_2 = tf.add(tf.layers.dropout(tf.matmul(self.layer_h_neg_2[h - 1], self.weights_domain2[h]),
                                   rate=self.keep_prob), self.bias_domain2[h])

            if h < self.num_cross_layers + 1:
                # project the representation of interaction to multiple latent spaces, generating multiple motivation representation
                multi_space_h_pos_1 = tf.matmul(self.layer_h_pos_1[h - 1], self.SWeight1[h])
                multi_space_h_pos_1 = tf.layers.dropout(multi_space_h_pos_1, rate=self.keep_prob)
                each_space_h_pos_1 = tf.split(multi_space_h_pos_1, self.fusion_size, axis=1)  # [1024, 4, 32]

                multi_space_h_neg_1 = tf.matmul(self.layer_h_neg_1[h - 1], self.SWeight1[h])
                multi_space_h_neg_1 = tf.layers.dropout(multi_space_h_neg_1, rate=self.keep_prob)
                each_space_h_neg_1 = tf.split(multi_space_h_neg_1, self.fusion_size, axis=1)

                multi_space_h_pos_2 = tf.matmul(self.layer_h_pos_2[h - 1], self.SWeight2[h])
                multi_space_h_pos_2 = tf.layers.dropout(multi_space_h_pos_2, rate=self.keep_prob)
                each_space_h_pos_2 = tf.split(multi_space_h_pos_2, self.fusion_size, axis=1)

                multi_space_h_neg_2 = tf.matmul(self.layer_h_neg_2[h - 1], self.SWeight2[h])
                multi_space_h_neg_2 = tf.layers.dropout(multi_space_h_neg_2, rate=self.keep_prob)
                each_space_h_neg_2 = tf.split(multi_space_h_neg_2, self.fusion_size, axis=1)



                # Given the knowledge of another domain, generating weights of attention
                # the shape of self.attentionWeight1[h]: [32*4*2, 4]
                PosAtten1 = tf.nn.softmax(tf.nn.relu(tf.layers.dropout(tf.matmul(
                    tf.concat([self.layer_h_pos_1[h - 1], multi_space_h_pos_1], axis=-1),
                    self.attentionWeight1_new[h]), rate=self.keep_prob)))
                NegAtten1 = tf.nn.softmax(tf.nn.relu(tf.layers.dropout(tf.matmul(
                    tf.concat([self.layer_h_neg_1[h - 1], multi_space_h_neg_1], axis=-1),
                    self.attentionWeight1_new[h]), rate=self.keep_prob)))
                PosAtten2 = tf.nn.softmax(tf.nn.relu(tf.layers.dropout(tf.matmul(
                    tf.concat([self.layer_h_pos_2[h - 1], multi_space_h_pos_2], axis=-1),
                    self.attentionWeight2_new[h]), rate=self.keep_prob)))
                NegAtten2 = tf.nn.softmax(tf.nn.relu(tf.layers.dropout(tf.matmul(
                    tf.concat([self.layer_h_neg_2[h - 1], multi_space_h_neg_2], axis=-1),
                    self.attentionWeight2_new[h]), rate=self.keep_prob)))

                #  Fuse the multiple representation with attention weights and achieve the cross-domain knowledge representation
                PosAtten1 = tf.split(PosAtten1, self.fusion_size, axis=1)
                after_fusion_h_pos_1 = tf.multiply(PosAtten1[0], each_space_h_pos_1[0])
                for i in range(1, self.fusion_size):
                    after_fusion_h_pos_1 += tf.multiply(PosAtten1[i], each_space_h_pos_1[i])
                after_fusion_h_pos_1 = tf.nn.l2_normalize(after_fusion_h_pos_1, axis=1)

                NegAtten1 = tf.split(NegAtten1, self.fusion_size, axis=1)
                after_fusion_h_neg_1 = tf.multiply(NegAtten1[0], each_space_h_neg_1[0])
                for i in range(1, self.fusion_size):
                    after_fusion_h_neg_1 += tf.multiply(NegAtten1[i], each_space_h_neg_1[i])
                after_fusion_h_neg_1 = tf.nn.l2_normalize(after_fusion_h_neg_1, axis=1)

                PosAtten2 = tf.split(PosAtten2, self.fusion_size, axis=1)
                after_fusion_h_pos_2 = tf.multiply(PosAtten2[0], each_space_h_pos_2[0])
                for i in range(1, self.fusion_size):
                    after_fusion_h_pos_2 += tf.multiply(PosAtten2[i], each_space_h_pos_2[i])
                after_fusion_h_pos_2 = tf.nn.l2_normalize(after_fusion_h_pos_2, axis=1)

                NegAtten2 = tf.split(NegAtten2, self.fusion_size, axis=1)
                after_fusion_h_neg_2 = tf.multiply(NegAtten2[0], each_space_h_neg_2[0])
                for i in range(1, self.fusion_size):
                    after_fusion_h_neg_2 += tf.multiply(NegAtten2[i], each_space_h_neg_2[i])
                after_fusion_h_neg_2 = tf.nn.l2_normalize(after_fusion_h_neg_2, axis=1)

                gate_weight_pos_1 = tf.sigmoid(tf.matmul(layer_h_pos_1, self.gateWeight_domain1[h]) +
                                               tf.matmul(after_fusion_h_pos_2, self.gateWeight_domain2[h]) +
                                               self.gateBias_domain1[h])
                layer_h_pos_1 = tf.add(tf.multiply(gate_weight_pos_1, layer_h_pos_1),
                                       tf.multiply((1-gate_weight_pos_1), after_fusion_h_pos_2))

                gate_weight_neg_1 = tf.sigmoid(tf.matmul(layer_h_neg_1, self.gateWeight_domain1[h]) +
                                               tf.matmul(after_fusion_h_neg_2, self.gateWeight_domain2[h]) +
                                               self.gateBias_domain1[h])
                layer_h_neg_1 = tf.add(tf.multiply(gate_weight_neg_1, layer_h_neg_1),
                                       tf.multiply((1 - gate_weight_neg_1), after_fusion_h_neg_2))

                gate_weight_pos_2 = tf.sigmoid(tf.matmul(layer_h_pos_2, self.gateWeight_domain2[h]) +
                                               tf.matmul(after_fusion_h_pos_1, self.gateWeight_domain1[h]) +
                                               self.gateBias_domain2[h])
                layer_h_pos_2 = tf.add(tf.multiply(gate_weight_pos_2, layer_h_pos_2),
                                       tf.multiply((1 - gate_weight_pos_2), after_fusion_h_pos_1))

                gate_weight_neg_2 = tf.sigmoid(tf.matmul(layer_h_neg_2, self.gateWeight_domain2[h]) +
                                               tf.matmul(after_fusion_h_neg_1, self.gateWeight_domain1[h]) +
                                               self.gateBias_domain2[h])
                layer_h_neg_2 = tf.add(tf.multiply(gate_weight_neg_2, layer_h_neg_2),
                                       tf.multiply((1 - gate_weight_neg_2), after_fusion_h_neg_1))

                Posterm1 = each_space_h_pos_1[0]
                PostermSquare1 = tf.square(each_space_h_pos_1[0])
                for i in range(1, self.fusion_size):
                    Posterm1 += each_space_h_pos_1[i]
                    PostermSquare1 += tf.square(each_space_h_pos_1[i])
                regu_inner_pos1 += layers.l2_regularizer(self.beta)(tf.square(Posterm1)-PostermSquare1)

                Negterm1 = each_space_h_neg_1[0]
                NegtermSquare1 = tf.square(each_space_h_neg_1[0])
                for i in range(1, self.fusion_size):
                    Negterm1 += each_space_h_neg_1[i]
                    NegtermSquare1 += tf.square(each_space_h_neg_1[i])
                regu_inner_neg1 += layers.l2_regularizer(self.beta)(tf.square(Negterm1)-NegtermSquare1)

                Posterm2 = each_space_h_pos_2[0]
                PostermSquare2 = tf.square(each_space_h_pos_2[0])
                for i in range(1, self.fusion_size):
                    Posterm1 += each_space_h_pos_2[i]
                    PostermSquare2 += tf.square(each_space_h_pos_2[i])
                regu_inner_pos2 += layers.l2_regularizer(self.beta)(tf.square(Posterm2) - PostermSquare2)

                Negterm2 = each_space_h_neg_2[0]
                NegtermSquare2 = tf.square(each_space_h_neg_2[0])
                for i in range(1, self.fusion_size):
                    Negterm2 += each_space_h_neg_2[i]
                    NegtermSquare2 += tf.square(each_space_h_neg_2[i])
                regu_inner_neg2 += layers.l2_regularizer(self.beta)(tf.square(Negterm2) - NegtermSquare2)

            if self.activation == 'relu':
                layer_h_pos_1 = tf.nn.relu(layer_h_pos_1)
                layer_h_neg_1 = tf.nn.relu(layer_h_neg_1)
                layer_h_pos_2 = tf.nn.relu(layer_h_pos_2)
                layer_h_neg_2 = tf.nn.relu(layer_h_neg_2)
            elif self.activation == 'sigmoid':
                layer_h_pos_1 = tf.nn.sigmoid(layer_h_pos_1)
                layer_h_neg_1 = tf.nn.sigmoid(layer_h_neg_1)
                layer_h_pos_2 = tf.nn.sigmoid(layer_h_pos_2)
                layer_h_neg_2 = tf.nn.sigmoid(layer_h_neg_2)
            self.layer_h_pos_1[h] = layer_h_pos_1
            self.layer_h_neg_1[h] = layer_h_neg_1
            self.layer_h_pos_2[h] = layer_h_pos_2
            self.layer_h_neg_2[h] = layer_h_neg_2

        self.pos_logits_1 = tf.squeeze(tf.matmul(layer_h_pos_1, self.h1) + self.b1)
        self.neg_logits_1 = tf.squeeze(tf.matmul(layer_h_neg_1, self.h1) + self.b1)

        self.pos_logits_2 = tf.squeeze(tf.matmul(layer_h_pos_2, self.h2) + self.b2)
        self.neg_logits_2 = tf.squeeze(tf.matmul(layer_h_neg_2, self.h2) + self.b2)

        reg_embedding_1 = layers.l2_regularizer(self.beta)(self.user_embeddings)
        reg_embedding_2 = layers.l2_regularizer(self.beta)(self.item_embeddings_1)
        reg_embedding_3 = layers.l2_regularizer(self.beta)(self.item_embeddings_1)

        reg_cost = reg_embedding_1 + reg_embedding_2 + reg_embedding_3 + \
                   regu_inner_pos1 + regu_inner_neg1 + regu_inner_pos2 + regu_inner_neg2

        cross_entropy_loss_1 = tf.reduce_mean(-tf.log(tf.sigmoid(self.pos_logits_1) + 1e-24)) + tf.reduce_mean(
            -tf.log(1 - tf.sigmoid(self.neg_logits_1) + 1e-24)) / 2

        cross_entropy_loss_2 = tf.reduce_mean(-tf.log(tf.sigmoid(self.pos_logits_2) + 1e-24)) + tf.reduce_mean(
            -tf.log(1 - tf.sigmoid(self.neg_logits_2) + 1e-24)) / 2

        self.loss = reg_cost + cross_entropy_loss_1 + cross_entropy_loss_2

        self.opt = tf.train.AdamOptimizer(learning_rate=lr)
        self.updates = self.opt.minimize(self.loss)


if __name__ == '__main__':
    print('begin to bulid D3CDR model using ' + params.metaName_1 + ' ' + params.metaName_2 + ' data')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    n_users_1, n_items_1, user_ratings_1 = load_data(filepath=params.filepath_1)
    n_users_2, n_items_2, user_ratings_2 = load_data(filepath=params.filepath_2)
    print(n_users_1, n_items_1)
    print(n_users_2, n_items_2)

    user_ratings_test_1 = generate_test(user_ratings_1)
    user_ratings_test_2 = generate_test(user_ratings_2)
    model = D3CDR(num_users=n_users_1, num_items_1=n_items_1, num_items_2=n_items_2,
                  num_cross_layers=num_cross, num_whole_layer=4, dim=32, layer_dim=[64, 32, 16, 8, 8], lr=params.LR, beta=0.01,
                  activation='relu', fusion_size=fusion_size)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    best_ret_1 = np.array([0] * 6)
    best_ret_2 = np.array([0] * 6)
    with tf.Session(config=config) as sess:
        if test:
            ret_1 = np.array([0.0] * 6)
            ret_2 = np.array([0.0] * 6)
            print("begin to test with pretrained model")
            saver = tf.train.Saver()
            saver.restore(sess, test_model)
            user_count_1 = 0
            user_count_2 = 0
            for t_uij in generate_test_batch(user_ratings_1, user_ratings_test_1, n_items_1):
                pos_s_1, neg_s_1 = sess.run([model.pos_logits_1, model.neg_logits_1],
                                            feed_dict={model.users_1: t_uij[:, 0],
                                                       model.pos_items_1: t_uij[:, 1],
                                                       model.neg_items_1: t_uij[:, 2]})
                user_count_1 += 1
                predictions_1 = [pos_s_1[0]]
                predictions_1 += list(neg_s_1)
                predictions_1 = [-1 * i for i in predictions_1]

                rank_1 = np.array(predictions_1).argsort().argsort()[0]
                if rank_1 < 5:
                    ret_1[0] += 1
                    ret_1[3] += 1 / np.log2(rank_1 + 2)
                if rank_1 < 10:
                    ret_1[1] += 1
                    ret_1[4] += 1 / np.log2(rank_1 + 2)
                if rank_1 < 20:
                    ret_1[2] += 1
                    ret_1[5] += 1 / np.log2(rank_1 + 2)

            print('%s: HR_5 %f HR_10 %f HR_20 %f'
                  % (params.metaName_1, ret_1[0] / user_count_1, ret_1[1] / user_count_1, ret_1[2] / user_count_1))
        else:
            print("session created")
            sess.run(tf.global_variables_initializer())
            print("-------------------initialization finished-------------------")
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)
            for k, v in zip(variable_names, values):
                print("Variable:", k)
                print("Shape: ", v.shape)
            start = 0
            model_saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
            bar_length = 300
            for epoch in range(start, params.N_EPOCH+1):
                ret_1 = np.array([0.0] * 6)
                ret_2 = np.array([0.0] * 6)
                cur_loss = 0
                widgets = ['Train: ', Percentage(), ' ', Bar('#'), ' ', ETA()]
                pbar = ProgressBar(widgets=widgets, maxval=bar_length).start()
                for i in range(bar_length):
                    pbar.update(i)
                    uij_1, uij_2 = generate_train_batch_for_all_overlap(user_ratings_1, user_ratings_test_1, n_items_1, 
                                                                        user_ratings_2, user_ratings_test_2, n_items_2, 
                                                                        batch_size=params.BATCH_SIZE)

                    feed_dict = {model.users_1: uij_1[:, 0],
                                 model.users_2: uij_2[:, 0],
                                 model.pos_items_1: uij_1[:, 1],
                                 model.pos_items_2: uij_2[:, 1],
                                 model.neg_items_1: uij_1[:, 2],
                                 model.neg_items_2: uij_2[:, 2],
                                 model.keep_prob: 0.5
                                 }
                    _, loss = sess.run([model.updates, model.loss], feed_dict=feed_dict)
                    cur_loss += loss
                pbar.finish()
                print('Epoch %d training loss %f' % (epoch, cur_loss))
                user_count_1 = 0
                user_count_2 = 0
                for t_uij_1, t_uij_2 in generate_test_batch_for_all_overlap(user_ratings_1, user_ratings_test_1, 
                                                                            n_items_1, user_ratings_2, 
                                                                            user_ratings_test_2, n_items_2):
                    pos_s_1, neg_s_1, pos_s_2, neg_s_2 = sess.run([model.pos_logits_1, model.neg_logits_1,
                                                                   model.pos_logits_2, model.neg_logits_2],
                                                                  feed_dict={model.users_1: t_uij_1[:, 0],
                                                                             model.users_2: t_uij_2[:, 0],
                                                                             model.pos_items_1: t_uij_1[:, 1],
                                                                             model.neg_items_1: t_uij_1[:, 2],
                                                                             model.pos_items_2: t_uij_2[:, 1],
                                                                             model.neg_items_2: t_uij_2[:, 2],
                                                                             model.keep_prob: 1
                                                                             })
                    user_count_1 += 1
                    predictions_1 = [pos_s_1[0]]
                    predictions_1 += list(neg_s_1)
                    predictions_1 = [-1 * i for i in predictions_1]

                    rank_1 = np.array(predictions_1).argsort().argsort()[0]
                    if rank_1 < 5:
                        ret_1[0] += 1
                        ret_1[3] += 1 / np.log2(rank_1 + 2)
                    if rank_1 < 10:
                        ret_1[1] += 1
                        ret_1[4] += 1 / np.log2(rank_1 + 2)
                    if rank_1 < 20:
                        ret_1[2] += 1
                        ret_1[5] += 1 / np.log2(rank_1 + 2)

                    user_count_2 += 1
                    predictions_2 = [pos_s_2[0]]
                    predictions_2 += list(neg_s_2)
                    predictions_2 = [-1 * i for i in predictions_2]

                    rank_2 = np.array(predictions_2).argsort().argsort()[0]
                    if rank_2 < 5:
                        ret_2[0] += 1
                        ret_2[3] += 1 / np.log2(rank_2 + 2)
                    if rank_2 < 10:
                        ret_2[1] += 1
                        ret_2[4] += 1 / np.log2(rank_2 + 2)
                    if rank_2 < 20:
                        ret_2[2] += 1
                        ret_2[5] += 1 / np.log2(rank_2 + 2)
                best_ret_1 = best_result(best_ret_1, ret_1)
                best_ret_2 = best_result(best_ret_2, ret_2)

                print('%s: HR_5 %f HR_10 %f HR_20 %f'
                      % (params.metaName_1, ret_1[0]/user_count_1, ret_1[1]/user_count_1, ret_1[2]/user_count_1))
                print('%s: NDCG_5 %f NDCG_10 %f NDCG_20 %f'
                      % (params.metaName_1, ret_1[3] / user_count_1, ret_1[4] / user_count_1, ret_1[5] / user_count_1))
                print('Best HitRatio for %s: HR_5 %f HR_10 %f HR_20 %f'
                      % (params.metaName_1, best_ret_1[0]/user_count_1, best_ret_1[1]/user_count_1,
                         best_ret_1[2]/user_count_1))
                print('Best NDCG for %s: NDCG_5 %f NDCG_10 %f NDCG_20 %f'
                      % (params.metaName_1, best_ret_1[3]/user_count_1, best_ret_1[4]/user_count_1,
                         best_ret_1[5]/user_count_1))

                if ret_1[0] == best_ret_1[0] or ret_1[1] == best_ret_1[1] or ret_1[2] == best_ret_1[2] \
                        or ret_1[3] == best_ret_1[3] or ret_1[4] == best_ret_1[4] or ret_1[5] == best_ret_1[5]:
                    save_name = params.MODEL_DIR + model_save_name
                    model_saver.save(sess, save_name, global_step=epoch + 1)
                    print("model-%s saved." % (epoch + 1))

                print('%s: HR_5 %f HR_10 %f HR_20 %f'
                      % (params.metaName_2, ret_2[0]/user_count_2, ret_2[1]/user_count_2, ret_2[2]/user_count_2))
                print('%s: NDCG_5 %f NDCG_10 %f NDCG_20 %f'
                      % (params.metaName_2, ret_2[3] / user_count_2, ret_2[4] / user_count_2, ret_2[5] / user_count_2))
                print('Best HitRatio for %s: HR_5 %f HR_10 %f HR_20 %f'
                      % (params.metaName_2, best_ret_2[0]/user_count_2, best_ret_2[1]/user_count_2,
                         best_ret_2[2]/user_count_2))
                print('Best NDCG for %s: NDCG_5 %f NDCG_10 %f NDCG_20 %f'
                      % (params.metaName_2, best_ret_2[3]/user_count_2, best_ret_2[4]/user_count_2,
                         best_ret_2[5]/user_count_2))

                if ret_2[0] == best_ret_2[0] or ret_2[1] == best_ret_2[1] or ret_2[2] == best_ret_2[2] \
                        or ret_2[3] == best_ret_2[3] or ret_2[4] == best_ret_2[4] or ret_2[5] == best_ret_2[5]:
                    save_name = params.MODEL_DIR + model_save_name
                    model_saver.save(sess, save_name, global_step=epoch + 1)
                    print("model-%s saved." % (epoch + 1))
