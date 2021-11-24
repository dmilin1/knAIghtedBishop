import chess
import numpy as np

class ChessUtils:

    def board_to_arr(board):
        return np.reshape(np.unpackbits(np.array([
            board.pieces(chess.PAWN, chess.WHITE).mask,
            board.pieces(chess.PAWN, chess.BLACK).mask,
            board.pieces(chess.ROOK, chess.WHITE).mask,
            board.pieces(chess.ROOK, chess.BLACK).mask,
            board.pieces(chess.KNIGHT, chess.WHITE).mask,
            board.pieces(chess.KNIGHT, chess.BLACK).mask,
            board.pieces(chess.BISHOP, chess.WHITE).mask,
            board.pieces(chess.BISHOP, chess.BLACK).mask,
            board.pieces(chess.QUEEN, chess.WHITE).mask,
            board.pieces(chess.QUEEN, chess.BLACK).mask,
            board.pieces(chess.KING, chess.WHITE).mask,
            board.pieces(chess.KING, chess.BLACK).mask
        ], dtype=np.uint64).view(np.uint8)), (12,8,8)).astype(np.float32)
    
    def get_value_arr(game, moves_left, total_move_count):
        moves_left_penalty = 0.1*(moves_left/total_move_count)**0.25
        return np.array(1.0 - moves_left_penalty if game.headers['Result'] == '1-0' else -1.0 + moves_left_penalty, dtype=np.float32)

    def get_policy_arr(move):
        arr = np.zeros(4096, dtype=np.float32)
        arr[ChessUtils.get_encoded_move(move)] = 1.0
        return arr
    
    def get_encoded_move(move):
        return move.from_square*64+move.to_square
    
    def mirror_move(move):
        return chess.Move(chess.square_mirror(move.from_square), chess.square_mirror(move.to_square))