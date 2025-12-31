import chess
from homemade import ExampleEngine
from chess.engine import Limit

engine = ExampleEngine()
board = chess.Board()

while not board.is_game_over():
    print(board)
    if board.turn == chess.WHITE:
        move = input("Your move (uci, e.g. e2e4): ")
        board.push_uci(move)
    else:
        bot_move = engine.choose_move(board, Limit(time=1.0))
        print("Bot plays:", bot_move)
        board.push(bot_move)

print(board.result())
