import sys
from lichess import Lichess
from bishopai import BishopAI
from brain import Brain

if len(sys.argv) <= 1 or sys.argv[1] == 'play':
    Lichess()
elif sys.argv[1] == 'train':
    brain = Brain()
    brain.learn(resume=' -n' not in ' '.join(sys.argv))
elif sys.argv[1] == 'test':
    bishopAI = BishopAI()
    # bishopAI.board.set_fen('r1bq1rk1/1ppnbppp/4pn2/3p4/1BPP4/P2BPN2/5PPP/RN1QK2R b KQ - 0 9')
    bishopAI.board.set_fen('rnbq1rk1/pp2ppbp/3p1np1/2p1P3/2PP1B2/5N2/PP1N1PPP/R2QKB1R b KQ - 0 7')
    # move = bishopAI.get_move()
    # bishopAI.catch_up([move])
    move = bishopAI.get_move()
else:
    print("""
        usage "py run.py <cmd> (-n)"
        -----
        cmd: (play, train, prune)
        -n: Build a new model instead of resuming. Overwrites current model. Can only be used with "train" cmd.
    """)