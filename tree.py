from re import search
import chess
import numpy as np
from chessUtils import ChessUtils

class Tree:

    C = 2 # A higher number results in more exploration
    CONCURRENT_ROLLOUT_DEPTH_WEIGHTING = 0.5

    def __init__(self, move=None, root=False):
        self.move = move
        self.historical = False if not root else True
        self.visited = False
        self.tree_size = 0
        self.tree_value_total = 0
        self.policy = 0
        self.branch_valuation = None
        self.puct = 0
        self.outcome = None
        self.parent = None
        self.children = {}
    
    def spawn_children(self, board, value, policy):
        legal_moves = list(board.legal_moves)
        encoded_legal_moves = np.array(list(map(lambda move: ChessUtils.get_encoded_move(move if board.turn else ChessUtils.mirror_move(move)), legal_moves)), dtype=np.uint32)
        self.elevate_data(1 - ( value + 1 ) / 2)
        normalized_policies = policy[encoded_legal_moves]
        normalized_policies /= sum(normalized_policies)
        for move, normalized_policy in zip(legal_moves, normalized_policies):
            child = Tree(move)
            child.parent = self
            child.policy = normalized_policy
            self.children[move.uci()] = child
            self.visited = True
    
    def mark_as_end_state(self, board):
        self.outcome = board.outcome()
        self.visited = True
        self.elevate_data(1)
    
    def elevate_data(self, value):
        self.tree_value_total += value
        self.tree_size += 1
        self.branch_valuation = self.tree_value_total / self.tree_size
        if self.parent and not self.historical:
            self.parent.elevate_data(1-value)
        
    def get_best_child(self):
        best_val = 0
        best_key = None
        for key, child in self.children.items():
            if child.branch_valuation is None:
                child.branch_valuation = 1 - self.branch_valuation
            puct = child.branch_valuation + Tree.C * child.policy * self.tree_size ** 0.5 / (child.tree_size + 1)
            child.puct = puct
            if puct > best_val:
                best_val = puct
                best_key = key
        return self.children[best_key]
    
    def get_most_explored_child(self):
        best_val = 0
        best_key = None
        for key, child in self.children.items():
            visit_count = child.tree_size
            if visit_count > best_val:
                best_val = visit_count
                best_key = key
        return self.children[best_key]
    
    def get_leaves(self, searchabilities=[], moves=[]):
        if self.visited == False:
            if not self.parent:
                return [[self, 0, []]]
            return [[self, np.average(searchabilities, weights=list(map(lambda n : n**Tree.CONCURRENT_ROLLOUT_DEPTH_WEIGHTING, range(len(searchabilities),0,-1)))), moves + [self.move]]]
        leaves = []
        for child in self.children.values():
            if child.branch_valuation is None:
                child.branch_valuation = 1 - self.branch_valuation
            child_leaves = child.get_leaves(searchabilities + [child.branch_valuation + Tree.C * child.policy * self.tree_size ** 0.5 / (child.tree_size + 1)], moves + ([self.move] if self.move else []))
            leaves += child_leaves
        return leaves

    
    def as_string(self, depth=0):
        my_str = f"{self.move.uci() if self.move else 'root'} - visits: {self.tree_size} - policy: {self.policy} - parent visits: {self.parent.tree_size if self.parent else 0} - valuation: {self.branch_valuation} - searchability: {self.puct}"
        if len(self.children):
            for i, child in enumerate(self.children.values()):
                if child.visited:
                    my_str += '\n' + ''.join(['   ' for _ in range(depth)]) + '|——' + child.as_string(depth=depth + 1)
        return my_str