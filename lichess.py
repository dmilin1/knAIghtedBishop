from dotenv import dotenv_values
import berserk
import threading
import sys
from bishopai import BishopAI


class Lichess:

    def __init__(self):
        try:
            self.session = berserk.TokenSession(dotenv_values(".env")['LICHESS_TOKEN'])
            self.client = berserk.Client(session=self.session)
        except KeyError:
            print('A Lichess token is required. Please set your "LICHESS_TOKN" environment variable.')
            quit()
        self.start_listening()
    
    def start_listening(self):
        print("awaiting challenges")
        for event in self.client.bots.stream_incoming_events():
            # print(event)
            if event['type'] == 'challenge':
                if Lichess.should_accept_challenge(event):
                    self.client.bots.accept_challenge(event['challenge']['id'])
                else:
                    self.client.bots.decline_challenge(event['challenge']['id'])
            elif event['type'] == 'gameStart':
                game = Game(self.client, event['game']['id'])
                game.start()
    
    def should_accept_challenge(event):
        if event['challenge']['variant']['key'] == 'standard':
            if event['challenge']['timeControl']['type'] == 'unlimited':
                return True
        return False


class Game(threading.Thread):

    def __init__(self, client, game_id, **kwargs):
        super().__init__(**kwargs)
        self.game_id = game_id
        self.client = client
        self.stream = client.bots.stream_game_state(game_id)
        self.bishopAI = BishopAI()
        self.is_white = True

    def run(self):
        print(f"Starting game: {self.game_id}")
        for event in self.stream:
            print(event)
            if event['type'] == 'gameStart': # just started
                self.handle_game_start(event)
            elif event['type'] == 'gameState': # move played
                self.handle_game_state(event)
            elif event['type'] == 'gameFull': # resuming game
                self.handle_game_full(event)
            elif event['type'] == 'chatLine': # chat event
                self.handle_chat_line(event)
    
    def handle_game_start(self, event):
        pass

    def handle_game_state(self, event):
        if event['status'] in ['resign', 'aborted']:
            print(f"Game ended with status {event['status']}: {self.game_id}")
            sys.exit()
        if event['moves']:
            self.bishopAI.catch_up([event['moves'].split(' ')[-1]])
        self.make_turn()

    def handle_game_full(self, event):
        if event['state']['moves']:
            self.bishopAI.catch_up(event['state']['moves'].split(' '))
        self.is_white = event['white']['id'] == 'bishopai'
        self.make_turn()

    def handle_chat_line(self, event):
        pass

    def make_turn(self):
        if self.bishopAI.board.turn == self.is_white:
            self.client.bots.make_move(self.game_id, self.bishopAI.get_move())