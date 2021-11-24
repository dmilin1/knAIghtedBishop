import numpy as np
from chessUtils import ChessUtils

class Tree:

    C = 1 # A higher number results in more exploration
    CONCURRENT_ROLLOUT_DEPTH_WEIGHTING = 0.5

    def __init__(self, move=None, root=False):
        self.move = move
        self.historical = False if not root else True
        self.visited = False
        self.cached = False
        self.tree_size = 0
        self.tree_value_total = 0
        self.tree_value = 0
        self.policy = 0
        self.branch_valuation = None
        self.puct = 0
        self.outcome = None
        self.endings_found = 0
        self.parent = None
        self.children = {}
    
    def spawn_children(self, board, value, policy):
        self.tree_value = value
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
    
    def get_best_leaf(self):
        if not self.visited or self.outcome:
            return self
        else:
            return self.get_best_child().get_best_leaf()
    
    def get_most_explored_child(self):
        best_val = 0
        best_key = None
        for key, child in self.children.items():
            visit_count = child.tree_size
            if visit_count > best_val:
                best_val = visit_count
                best_key = key
        return self.children[best_key]
    
    def get_siblings(self):
        if self.parent is None:
            return []
        else:
            return [child for child in self.parent.children.values() if child is not self]
        

    def get_all_leaves(self):
        if not self.visited and not self.historical:
            return [self]
        else:
            leaves = []
            for child in self.children.values():
                leaves += child.get_all_leaves()
            return leaves
    
    def as_string(self, recurse=False, visitedOnly=False, depth=0):
        my_str = f"{self.move.uci() if self.move else 'root'} - visits: {self.tree_size} - policy: {self.policy} - parent visits: {self.parent.tree_size if self.parent else 0} - valuation: {self.branch_valuation} - searchability: {self.puct}"
        if (recurse or depth < 1) and len(self.children):
            for i, child in enumerate(self.children.values()):
                if child.visited or not visitedOnly:
                    my_str += '\n' + ''.join(['   ' for _ in range(depth)]) + '|——' + child.as_string(recurse, visitedOnly, depth + 1)
        return my_str
    
    def get_move_chain(parent, child):
        if child == parent or child.parent is None:
            return []
        else:
            return Tree.get_move_chain(parent, child.parent) + [child.move]
    