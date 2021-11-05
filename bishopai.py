import chess
from chessUtils import ChessUtils
import numpy as np
from tree import Tree
from brain import Brain
from timer import Timer

class BishopAI:

    CONCURRENT_ROLLOUT_SIZE = 64

    def __init__(self):
        self.brain = Brain(load_saved=True)
        self.board = chess.Board()
        self.tree = Tree(root=True)
    
    def catch_up(self, moves):
        for move in moves:
            self.board.push_san(move)
            if self.tree.children and move in self.tree.children:
                self.tree = self.tree.children[move]
                self.tree.historical = True
            else:
                self.tree = Tree(root=True)
                print('forced to reset tree in memory')
            print(self)

    def get_move(self):
        # for i in range(30):
        #     self.perform_concurrent_simulations(self.tree)
        # for child in self.tree.children.values():
        #     print(f"{child.move.uci()} - visits: {child.tree_size} - policy: {child.policy} - parent visits: {self.tree.tree_size} - valuation: {child.branch_valuation} - searchability: {child.puct}")

        for i in range(600):
            if i % 20 == 0:
                print(f"\rthinking: {i}", end='')
            # Timer.start('simulation')
            self.perform_simulation(self.tree)
            # Timer.stop('simulation')
        print()
        for child in self.tree.children.values():
            print(f"{child.move.uci()} - visits: {child.tree_size} - policy: {child.policy} - parent visits: {self.tree.tree_size} - valuation: {child.branch_valuation} - searchability: {child.puct}")
        # print()
        # print(self.tree.as_string())
        Timer.print()
        return self.tree.get_most_explored_child().move.uci()
    
    def perform_concurrent_simulations(self, tree):
        leaves = tree.get_leaves()
        leaves.sort(key=lambda leaf_bundle : leaf_bundle[1], reverse=True)
        leaves = leaves[0: min(len(leaves), BishopAI.CONCURRENT_ROLLOUT_SIZE)]
        boards = []
        for leaf, score, moves in leaves:
            for move in moves:
                self.board.push(move)
            boards.append(ChessUtils.board_to_arr(self.board if self.board.turn else self.board.mirror()))
            for move in moves:
                self.board.pop()
        values, policies = self.brain.predict_raw(np.array(boards))
        for i, leaf_bundle in enumerate(leaves):
            leaf, score, moves = leaf_bundle
            for move in moves:
                self.board.push(move)
            leaf.spawn_children(self.board, values[i][0], policies[i])
            for move in moves:
                self.board.pop()


    def perform_simulation(self, tree):
        if not tree.historical:
            self.board.push(tree.move)
        if not tree.visited:
            if self.board.is_game_over():
                tree.mark_as_end_state(self.board)
            else:
                # Timer.start('prediction')
                value, policy = self.brain.predict(self.board if self.board.turn else self.board.mirror())
                # Timer.stop('prediction')
                tree.spawn_children(self.board, value[0][0], policy[0])
        elif tree.outcome:
            tree.elevate_data(1)
        else:
            self.perform_simulation(tree.get_best_child())
        if tree.move is not None and not tree.historical:
            self.board.pop()
    
    def __str__(self):
        return f"\n{self.board}\n"