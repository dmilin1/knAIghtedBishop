import sys
from lichess import Lichess
from bishopai import BishopAI
from brain import Brain

if len(sys.argv) <= 1 or sys.argv[1] == 'play':
    Lichess()
elif sys.argv[1] == 'train':
    brain = Brain()
    brain.learn()
elif sys.argv[1] == 'test':
    bishopAI = BishopAI()
    bishopAI.board.set_fen('r1bq1rk1/1ppnbppp/4pn2/3p4/1BPP4/P2BPN2/5PPP/RN1QK2R b KQ - 0 9')
    bishopAI.get_move()
else:
    print("""
        usage "py run.py <cmd>"
        cmd: (play, train)
    """)