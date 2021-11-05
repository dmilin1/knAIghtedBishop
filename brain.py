import chess
import chess.pgn
import numpy as np
from tensorflow import keras
from keras import layers, regularizers
from chessUtils import ChessUtils
import random
import sys

class Brain:

    BATCH_SIZE = 1024
    SAMPLE_MEM_SIZE = 100000 # individual board states to hold in memory while shuffling

    def __init__(self, load_saved=True):
        if load_saved:
            self.load_model()

    def gen_game_data():
        pgn = open("D:/Chess Datasets/big.pgn")
        
        while True:
            pointer = pgn.tell()
            headers = chess.pgn.read_headers(pgn)
            if (
                'Bullet' not in headers['Event']
                and (headers['Result'] == "1-0" or headers['Result'] == "0-1")
                and (int(headers['WhiteElo']) > 1800 and int(headers['BlackElo']) > 1800)
                # and (int(headers['WhiteElo']) < 1200 and int(headers['BlackElo']) < 1200)
                and abs(int(headers['WhiteElo']) - int(headers['BlackElo'])) < 100
                and headers['Termination'] == 'Normal'
            ):
                pgn.seek(pointer)
                yield chess.pgn.read_game(pgn)
            else:
                continue
    
    def gen_training_data():
        samples = []
        game_gen = Brain.gen_game_data()

        # pretrain on some random data because it does something for some reason
        # for i in range(1):
        #     yield (np.random.rand(Brain.BATCH_SIZE, 12, 8, 8), [np.random.rand(Brain.BATCH_SIZE), np.random.rand(Brain.BATCH_SIZE, 4096)])

        chunk_num = 0
        while True:
            chunk_num += 1
            rand_list = np.random.rand(10000)
            while len(samples) < Brain.SAMPLE_MEM_SIZE:
                game = next(game_gen)
                board = game.board()
                for i, move in enumerate(game.mainline_moves()):
                    if (rand_list[i] < 0.1 and i < 10) or rand_list[i] < 0.25:
                        inputs = ChessUtils.board_to_arr(board if board.turn else board.mirror())
                        value = ChessUtils.get_value_arr(game)
                        policy = ChessUtils.get_policy_arr(move if board.turn else ChessUtils.mirror_move(move))
                        samples.append([inputs, value if board.turn else -value, policy])
                    board.push(move)
            
            random.shuffle(samples)
            data = np.array(samples[0:Brain.BATCH_SIZE], dtype=object)
            samples = samples[Brain.BATCH_SIZE:]
            inputs_chunk = np.array([x[0] for x in data], dtype=np.float32)
            value_chunk = np.array([x[1] for x in data], dtype=np.float32)
            policy_chunk = np.array([x[2] for x in data], dtype=np.float32)
            yield (inputs_chunk, [value_chunk, policy_chunk])

    def build_model(self):
        kernel_reg=0.00001
        seed=1
        kernel_init=keras.initializers.glorot_uniform(seed=seed)
        kernel_init_dense=keras.initializers.glorot_uniform(seed=seed)

        x_in = x = layers.Input((12,8,8))
        x = layers.Conv2D(256, (5,5), kernel_regularizer=regularizers.l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(x)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)


        def residualLayers(inputLayering, count):
            if (count == 0):
                return inputLayering

            x = layers.Conv2D(256, (3,3), kernel_regularizer=regularizers.l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(inputLayering)
            x = layers.Activation('relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(256, (3,3), kernel_regularizer=regularizers.l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(inputLayering)
            x = layers.Add()([x, inputLayering])
            x = layers.Activation('relu')(x)
            x = layers.BatchNormalization()(x)
            return residualLayers(x, count - 1)


        resid = residualLayers(x, 7)

        x = layers.Conv2D(4, (8,8), kernel_regularizer=regularizers.l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(resid)
        x = layers.Activation('relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(768, use_bias=False, kernel_regularizer=regularizers.l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(512, use_bias=False, kernel_regularizer=regularizers.l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)
        x = layers.Activation('relu')(x)
        x = layers.Dense(1, kernel_initializer=kernel_init_dense)(x)

        value_out = layers.Activation('tanh', name='value')(x)


        def policy(inputLayering, name):
            x = layers.Conv2D(8, (8,8), kernel_regularizer=regularizers.l2(kernel_reg), kernel_initializer=kernel_init, padding='same', data_format='channels_first', use_bias=False)(inputLayering)
            x = layers.Activation('relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Flatten()(x)
            x = layers.Dense(512, use_bias=False, kernel_regularizer=regularizers.l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)
            x = layers.Activation('relu')(x)
            x = layers.Dense(4096, use_bias=False, kernel_regularizer=regularizers.l2(kernel_reg), kernel_initializer=kernel_init_dense)(x)
            return layers.Activation('softmax', name=name)(x)

        policy_out = policy(resid, 'policy')
        self.model = keras.Model(x_in, [value_out, policy_out])
        print(self.model.summary())

        self.model.compile(loss=['mean_squared_error','categorical_crossentropy'],
        optimizer=keras.optimizers.Adam(),
        metrics={
            'value': keras.metrics.mean_absolute_error,
            'policy': keras.metrics.categorical_accuracy,
        })

    def load_model(self):
        self.model = keras.models.load_model(f"{sys.path[0]}/models/model")

    def learn(self):
        self.build_model()
        self.model.fit(
            Brain.gen_training_data(),
            epochs=1000,
            steps_per_epoch=1000,
            callbacks=[
                SaveModel()
            ]
        )

    def predict(self, board):
        if not self.model:
            self.load_model()
        return self.model.predict(np.array([ChessUtils.board_to_arr(board)]))
    
    def predict_raw(self, boards):
        if not self.model:
            self.load_model()
        return self.model.predict(boards)

class SaveModel(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.save(f"{sys.path[0]}/models/model")