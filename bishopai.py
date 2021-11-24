import chess
from chessUtils import ChessUtils
import numpy as np
from tree import Tree
from brain import Brain
from timer import Timer

class BishopAI:

    CONCURRENT_ROLLOUT_BATCH_SIZE = 1
    ROLLOUTS_PER_MOVE = 400

    def __init__(self):
        self.brain = Brain(load_saved=True)
        self.board = chess.Board()
        self.tree = Tree(root=True)
        self.cache_hits = 0
        self.cache_misses = 0
    
    def catch_up(self, moves):
        print(self.board)
        for move in moves:
            self.board.push_san(move)
            if self.tree.children and move in self.tree.children:
                self.tree = self.tree.children[move]
                self.tree.historical = True
            else:
                self.tree = Tree(root=True)
                print('forced to reset tree in memory')
            print(self)
    
    def add_moves(self, moves):
        for move in moves:
            self.board.push(move)

    def remove_moves(self, moves):
        for move in moves:
            self.board.pop()

    def get_move(self):
        Timer.start('run_concurrent')
        for i in range(BishopAI.ROLLOUTS_PER_MOVE):
            if i % 20 == 0:
                print(f"\rthinking: {i}", end='')
            self.perform_concurrent_simulations(self.tree, i)
        selected_move = self.tree.get_most_explored_child().move.uci()
        print('\n\n' + f"{selected_move} is best move with victory odds of {round(self.tree.children[selected_move].branch_valuation * 100, 2)}%")
        print(f"{self.tree.endings_found} end games explored")
        print('\n' + self.tree.as_string(recurse=False, visitedOnly=True))
        print(f"Cache hit rate {100*self.cache_hits/(self.cache_hits + self.cache_misses)}% ({self.cache_hits}/{self.cache_hits + self.cache_misses})")
        Timer.stop('run_concurrent')
        Timer.print()

        # stored_fen = self.board.fen()
        # self.__init__()
        # self.board.set_fen(stored_fen)

        # Timer.start('run_single')
        # for i in range(BishopAI.ROLLOUTS_PER_MOVE):
        #     if i % 20 == 0:
        #         print(f"\rthinking: {i}", end='')
        #     self.perform_simulation(self.tree)
        # selected_move = self.tree.get_most_explored_child().move.uci()
        # print('\n' + self.tree.as_string(recurse=False, visitedOnly=True))
        # Timer.stop('run_single')
        # Timer.print()
        return selected_move
    
    def perform_concurrent_simulations(self, tree, sim_num):
        best_leaf = tree.get_best_leaf()
        if best_leaf.outcome:
            best_leaf.elevate_data(1)
            tree.endings_found += 1
            return
        move_chain = Tree.get_move_chain(tree, best_leaf)
        self.add_moves(move_chain)
        if self.board.is_game_over():
            best_leaf.mark_as_end_state(self.board)
            self.remove_moves(move_chain)
            return
        best_leaf_key = BrainCache.queue_for_prediction(self.board if self.board.turn else self.board.mirror())
        if not BrainCache.is_cached(best_leaf_key):
            self.remove_moves(move_chain)
            Timer.start('searching_to_cache')
            if BishopAI.CONCURRENT_ROLLOUT_BATCH_SIZE > 1:
                self.cache_good_neighbors(best_leaf, BishopAI.CONCURRENT_ROLLOUT_BATCH_SIZE) # max(2, 2 ** int(5 * (( 600 - sim_num) / 600)))
            Timer.stop('searching_to_cache')
            Timer.start('predicting')
            BrainCache.predict_on_queued(self.brain)
            Timer.stop('predicting')
            self.add_moves(move_chain)
            self.cache_misses += 1
        else:
            self.cache_hits += 1
        value, policy = BrainCache.get_cached(best_leaf_key)
        best_leaf.spawn_children(self.board, value, policy)
        self.remove_moves(move_chain)
    
    def cache_good_neighbors(self, best_leaf, num_to_cache):
        potential_neighbors = []
        current_tree = self.tree
        move_path = Tree.get_move_chain(self.tree, best_leaf)
        for i, move in enumerate(move_path):
            current_tree = current_tree.children[move.uci()]
            for neighbor in current_tree.get_siblings():
                if not neighbor.visited and not neighbor.cached:
                    neighbor.cache_value = abs(current_tree.puct - neighbor.puct)
                    potential_neighbors.append(neighbor)
        potential_neighbors.sort(key = lambda neighbor : neighbor.cache_value, reverse=False)
        for neighbor in potential_neighbors[0:num_to_cache-1]:
            neighbor.cached = True
            move_path = Tree.get_move_chain(self.tree, neighbor)
            self.add_moves(move_path)
            BrainCache.queue_for_prediction(self.board if self.board.turn else self.board.mirror())
            self.remove_moves(move_path)

    def perform_simulation(self, tree):
        if not tree.historical:
            self.board.push(tree.move)
        if not tree.visited:
            if self.board.is_game_over():
                tree.mark_as_end_state(self.board)
            else:
                value, policy = self.brain.predict(self.board if self.board.turn else self.board.mirror())
                tree.spawn_children(self.board, value[0][0], policy[0])
        elif tree.outcome:
            tree.elevate_data(1)
        else:
            self.perform_simulation(tree.get_best_child())
        if tree.move is not None and not tree.historical:
            self.board.pop()
    
    def __str__(self):
        return f"\n{self.board}\n"


class BrainCache:

    cache = {}
    queued = {}

    def queue_for_prediction(board):
        key = board.fen()
        if key not in BrainCache.queued and key not in BrainCache.cache:
            BrainCache.queued[key] = ChessUtils.board_to_arr(board)
        return key

    def predict_on_queued(brain):
        if len(BrainCache.queued) == 0:
            return
        values, policies = brain.predict_raw(np.array(list(BrainCache.queued.values())))
        for i, key in enumerate(BrainCache.queued):
            BrainCache.cache[key] = (values[i][0], policies[i])
        BrainCache.queued = {}
    
    def get_cached(key):
        return BrainCache.cache[key]

    def is_cached(key):
        return key in BrainCache.cache