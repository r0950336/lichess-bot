import os
import sys
import time
import json
import random
import logging
from dataclasses import dataclass

# Add project root to sys.path so "import lib" works when launched from /engines
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import chess
from chess.engine import PlayResult, Limit
from lib.engine_wrapper import MinimalEngine
from lib.lichess_types import MOVE, HOMEMADE_ARGS_TYPE

logger = logging.getLogger(__name__)

# Config
# Path to the JSON file that stores the opening book
BOOK_PATH = "opening_book.json"

# Maximum number of plies (half-moves) where the opening book is allowed
# After this, the engine relies purely on search
BOOK_MAX_PLIES = 16

# Minimum number of games required for a book move to be trusted
# Prevents using rare / noisy book moves
BOOK_MIN_N = 8

# Maximum depth for the alpha-beta search
MAX_DEPTH = 6

# Default thinking time per move (used if no clock info is available)
DEFAULT_BUDGET_SEC = 1.5  # safe default

# A very large number used as "infinity" in alpha-beta
INFINITY = 10**9

# Score used to represent checkmate (must be lower than INFINITY)
MATE_SCORE = 10**8

# Simple opening book (optional)
def load_opening_book(path: str = BOOK_PATH) -> dict:
    """
    Loads the opening book from a JSON file.
    Returns a dictionary mapping FEN positions to possible moves.
    """

    try:
        # Open the opening book file
        with open(path, "r", encoding="utf-8") as f:
            # Load JSON and extract only the "book" section
            return json.load(f).get("book", {})

    except FileNotFoundError:
        # If the book file does not exist, warn and continue without it
        logger.warning("No opening_book.json found. Playing without opening book.")
        return {}

    except Exception as e:
        # Catch any other errors (corrupt JSON, permission issues, etc.)
        logger.warning("Failed to load opening book: %s", e)
        return {}

# Load the opening book once at engine startup
OPENING_BOOK = load_opening_book()

def pick_book_move(board: chess.Board) -> chess.Move | None:
    """
    Picks the best opening-book move for the current position.
    Returns None if no suitable book move exists.
    """

    # Look up the current position using its FEN string
    moves = OPENING_BOOK.get(board.fen())

    # If this position is not in the book, give up
    if not moves:
        return None

    # Best move found so far (as UCI string)
    best_uci = None

    # Comparison key: (average score, number of games)
    # Start with very bad defaults so any real move beats it
    best_key = (-999.0, -1)

    # Iterate through all candidate book moves
    for uci, info in moves.items():

        # Number of games this move was played
        n = int(info.get("n", 0))

        # Skip moves that don't have enough data
        if n < BOOK_MIN_N:
            continue

        # Average result of this move (e.g. win rate or score)
        avg = float(info.get("avg", 0.0))

        # Use (avg, n) so higher avg wins, ties broken by more games
        key = (avg, n)

        # If this move is better than the previous best, store it
        if key > best_key:
            best_key = key
            best_uci = uci

    # If no move survived filtering, return None
    if not best_uci:
        return None

    # Convert the UCI string to a chess.Move object
    mv = chess.Move.from_uci(best_uci)

    # Only return the move if it is still legal in the current position
    return mv if mv in board.legal_moves else None

# Evaluation
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def capture_is_obviously_bad(board: chess.Board, mv: chess.Move) -> bool:
    """
    Very cheap SEE-lite:
    If it's a capture and the capturing piece will be taken back immediately
    and we didn't win enough value, skip.
    """
    if not board.is_capture(mv):
        return False

    victim = board.piece_at(mv.to_square)
    attacker = board.piece_at(mv.from_square)
    if not victim or not attacker:
        return False

    v = PIECE_VALUES.get(victim.piece_type, 0)
    a = PIECE_VALUES.get(attacker.piece_type, 0)

    board.push(mv)

    # if opponent can recapture the attacker immediately
    can_recapture = any(
        board.is_capture(rm) and rm.to_square == mv.to_square
        for rm in board.legal_moves
    )

    board.pop()

    # If we win a pawn but give a bishop/rook/queen: obviously bad
    if can_recapture and v + 50 < a:
        return True

    return False


# PST (white perspective) – modest, not extreme
PST_PAWN = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-10,-10, 10, 10,  5,
     5, -5, -5,  5,  5, -5, -5,  5,
     0,  0,  0, 10, 10,  0,  0,  0,
     5,  5, 10, 20, 20, 10,  5,  5,
    10, 10, 20, 25, 25, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
]
PST_KNIGHT = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
]
PST_BISHOP = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
]
PST_ROOK = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
]
PST_QUEEN = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -10,  0,  5,  5,  5,  5,  0,-10,
    -5,  0,  5,  5,  5,  5,  0, -5,
     0,  0,  5,  5,  5,  5,  0, -5,
   -10,  5,  5,  5,  5,  5,  0,-10,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
]
PST_KING_MID = [
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -10,-20,-20,-20,-20,-20,-20,-10,
    20, 20,  0,  0,  0,  0, 20, 20,
    20, 30, 10,  0,  0, 10, 30, 20,
]

PST = {
    chess.PAWN: PST_PAWN,        # pawn positional adjustments
    chess.KNIGHT: PST_KNIGHT,    # knight centralization adjustments
    chess.BISHOP: PST_BISHOP,    # bishop activity adjustments
    chess.ROOK: PST_ROOK,        # rook file/rank activity adjustments
    chess.QUEEN: PST_QUEEN,      # queen activity/safety adjustments
    chess.KING: PST_KING_MID,    # king safety adjustments for middlegame
}

def _pst_bonus(piece: chess.Piece, sq: int) -> int:
    # Look up the piece-square table (PST) for this piece type (pawn/knight/...)
    table = PST.get(piece.piece_type)

    # If we don't have a PST for this piece type, no positional bonus applies
    if table is None:
        return 0

    # PST is defined from White's perspective.
    # So for Black pieces we mirror the square so the same table "works" for them too.
    idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)

    # Return the positional bonus/penalty for this piece on that (possibly mirrored) square
    return table[idx]


def king_safety_score(board: chess.Board, color: bool) -> int:
    score = 0  # accumulator for king safety heuristics

    ksq = board.king(color)  # get the king square for the requested color
    if ksq is None:
        return 0  # should not happen in legal chess, but safe guard for weird positions

    # Consider "queens still on the board" as a proxy for middlegame danger.
    # This becomes True only if BOTH sides still have a queen.
    queens_on = (len(board.pieces(chess.QUEEN, chess.WHITE)) > 0 and
                 len(board.pieces(chess.QUEEN, chess.BLACK)) > 0)

    # Castled squares bonus:
    # Encourage castling by giving a big bonus when king is on the typical castled squares.
    if color == chess.WHITE and ksq in (chess.G1, chess.C1):  # white king on short/long castle squares
        score += 80  # reward king safety / rook activation
    if color == chess.BLACK and ksq in (chess.G8, chess.C8):  # black king on short/long castle squares
        score += 80  # same idea for black

    # Penalize king wandering while queens exist:
    # When queens are on, the king is much easier to get checkmated if it moves away from home/castle squares.
    if queens_on:
        # For White, allow the starting square E1 and castled squares G1/C1 without penalty.
        if color == chess.WHITE and ksq not in (chess.E1, chess.G1, chess.C1):
            score -= 140  # strong penalty to discourage early king walks
        # For Black, allow E8/G8/C8 without penalty.
        if color == chess.BLACK and ksq not in (chess.E8, chess.G8, chess.C8):
            score -= 140  # same strong penalty for black

    # Being in check:
    # `board.is_check()` tells if the side to move is currently in check.
    # We only penalize if the side-to-move is the same as `color`,
    # otherwise we'd be penalizing the wrong king.
    if board.is_check() and board.turn == color:
        score -= 60  # moderate penalty for allowing checks / unsafe king

    return score  # final king safety evaluation component


def hangs_major_piece_after(board: chess.Board) -> bool:
    """
    Call AFTER board.push(mv). Now it's opponent to move.
    Return True if opponent has a legal capture of our queen or rook.
    """
    my_color = not board.turn  # after pushing our move, board.turn is opponent; so "we" are the opposite
    # Collect all our queens and rooks (major pieces) as target squares to protect
    targets = set(board.pieces(chess.QUEEN, my_color)) | set(board.pieces(chess.ROOK, my_color))

    if not targets:
        return False  # if we have no queen/rooks left, there's nothing "major" to hang

    # Iterate over every legal reply the opponent has
    for m in board.legal_moves:
        # If opponent's move is a capture and it lands on one of our queen/rook squares -> we hung a major piece
        if board.is_capture(m) and m.to_square in targets:
            return True  # immediately report danger

    return False  # no immediate capture of our queen/rook was found


def opponent_has_mate_in_1(board: chess.Board) -> bool:
    """
    Assumes it's opponent to move (i.e., you already did board.push(mv)).
    Returns True if opponent has any move that checkmates immediately.
    """
    # For every legal move available to the side to move (the opponent)
    for r in board.legal_moves:
        board.push(r)  # make the move on the board
        is_mate = board.is_checkmate()  # check if that move delivers checkmate
        board.pop()  # undo the move to restore original position
        if is_mate:
            return True  # opponent can mate immediately, so our previous move was a blunder
    return False  # no mate-in-1 found for the opponent
    """
    Assumes it's opponent to move (i.e., you already did board.push(mv)).
    Returns True if opponent has any move that checkmates immediately.
    """
    for r in board.legal_moves:
        board.push(r)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return True
    return False


def safe_opening_move(board: chess.Board) -> chess.Move | None:
    # If we are in check, we must respond tactically; do not "autopilot" an opening move
    if board.is_check():
        return None

    # If there is any capture available, it might be tactical (threats, hanging pieces),
    # so we avoid blindly playing a preset opening move.
    if any(board.is_capture(m) for m in board.legal_moves):
        return None  # tactics: don't autopilot

    # A small hardcoded list of common sensible opening moves (UCI strings),
    # chosen depending on whose turn it is.
    preferred = (
        ["e2e4", "d2d4", "c2c4", "g1f3", "b1c3", "f1c4", "e1g1"] if board.turn == chess.WHITE
        else ["e7e5", "d7d5", "c7c5", "g8f6", "b8c6", "g7g6", "e8g8"]
    )

    # Try each preferred move in order and pick the first one that is legal in the current position
    for uci in preferred:
        mv = chess.Move.from_uci(uci)  # convert "e2e4" -> chess.Move object
        if mv in board.legal_moves:  # legality check against current position
            return mv  # found a safe, legal opening move

    return None  # no preferred move was legal in this position


def queen_can_be_trapped_soon(board: chess.Board) -> bool:
    # This is meant to be called AFTER a push, so it's now opponent to move.
    # Therefore "my_color" is the side that just moved.
    my_color = not board.turn

    qs = list(board.pieces(chess.QUEEN, my_color))  # list all our queen squares
    if not qs:
        return False  # no queen exists, so it can't be trapped

    qsq = qs[0]  # take the (only) queen's square (normal chess has max 1 queen unless promotions)
    attacked = board.is_attacked_by(board.turn, qsq)  # is the opponent currently attacking our queen square?

    # Rough mobility: count how many squares the queen *could* move to (attacks),
    # excluding squares occupied by our own pieces (since those block legal destinations).
    q_moves = 0  # mobility counter
    for sq in board.attacks(qsq):  # all squares the queen attacks from its square (ray-based)
        p = board.piece_at(sq)  # see what's on that square (if anything)
        if p is None or p.color != my_color:  # empty square OR enemy piece (both are potential destinations)
            q_moves += 1  # count it as a "way out" / mobility

    # If queen is attacked AND has very limited mobility, flag as "could be trapped soon"
    # (threshold 2 is very strict; it will only trigger in pretty severe cases)
    return attacked and q_moves <= 2  # True means: attacked + almost no escape squares


def is_passed_pawn(board: chess.Board, sq: int, color: bool) -> bool:
    # Determine the pawn's file (0..7 for a..h)
    file = chess.square_file(sq)

    # Determine the pawn's rank (0..7 for rank 1..8 in python-chess indexing)
    rank = chess.square_rank(sq)

    # White pawns move "up" in rank (+1), black pawns move "down" in rank (-1)
    direction = 1 if color == chess.WHITE else -1

    # A pawn is "passed" if there are NO enemy pawns in front of it on:
    # - same file
    # - adjacent files (left/right)
    for df in (-1, 0, 1):  # check file-1, file, file+1
        f = file + df  # candidate file to scan
        if f < 0 or f > 7:
            continue  # skip off-board files

        # Start scanning squares *in front* of the pawn on this file
        r = rank + direction
        while 0 <= r <= 7:  # walk forward until we hit the last rank
            s = chess.square(f, r)  # build a square from file f and rank r
            p = board.piece_at(s)   # get whatever piece is on that square (if any)

            # If we see an enemy pawn ahead on same/adjacent file, the pawn is NOT passed
            if p and p.piece_type == chess.PAWN and p.color != color:
                return False  # blocked by an opposing pawn in its "passing lane"

            r += direction  # move further forward along the file

    return True  # no enemy pawns found in front on same/adjacent files -> passed pawn    file = chess.square_file(sq)
    rank = chess.square_rank(sq)
    direction = 1 if color == chess.WHITE else -1

    for df in (-1, 0, 1):
        f = file + df
        if f < 0 or f > 7:
            continue
        r = rank + direction
        while 0 <= r <= 7:
            s = chess.square(f, r)
            p = board.piece_at(s)
            if p and p.piece_type == chess.PAWN and p.color != color:
                return False
            r += direction
    return True


def passed_bonus(sq: int, color: bool) -> int:
    # Convert square rank into "progress rank" from the pawn's own perspective:
    # - for White: higher rank index means closer to promotion
    # - for Black: mirror the rank so that moving downward still increases "progress"
    r = chess.square_rank(sq) if color == chess.WHITE else (7 - chess.square_rank(sq))

    # Give bigger bonus the closer the pawn is to promotion
    if r >= 6: return 300  # very close (typically 7th rank from its perspective) -> huge bonus
    if r == 5: return 140  # close -> strong bonus
    if r == 4: return 70   # advanced -> moderate bonus
    return 20              # passed but not far advanced -> small baseline bonus    r = chess.square_rank(sq) if color == chess.WHITE else (7 - chess.square_rank(sq))
    if r >= 6: return 300
    if r == 5: return 140
    if r == 4: return 70
    return 20


def evaluate(board: chess.Board) -> int:
    # -----------------------------
    # Terminal checks (game over)
    # -----------------------------

    # If side to move is checkmated, evaluation should be very bad for them
    # (you return -MATE_SCORE because you score from side-to-move perspective)
    if board.is_checkmate():
        return -MATE_SCORE

    # Stalemate or not enough material is a draw -> neutral evaluation
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    # "stm" = side to move (the perspective you evaluate from)
    stm = board.turn

    score = 0  # main evaluation accumulator

    # -----------------------------
    # 1) MATERIAL + PST
    # -----------------------------
    # Loop over all pieces currently on the board:
    # board.piece_map() returns {square: Piece, ...}
    for sq, piece in board.piece_map().items():
        val = PIECE_VALUES[piece.piece_type]  # base material value (e.g., pawn=100, knight=320, ...)
        pst = _pst_bonus(piece, sq)           # positional tweak from piece-square table

        # Add if it's our piece (side to move), subtract if it's opponent's
        # This makes evaluation always "good for side to move = positive".
        if piece.color == stm:
            score += val + pst
        else:
            score -= val + pst

        # -----------------------------
        # passed pawn bonus (extra)
        # -----------------------------
        # If the piece is a pawn and it's passed, give a bonus that grows with advancement.
        if piece.piece_type == chess.PAWN and is_passed_pawn(board, sq, piece.color):
            b = passed_bonus(sq, piece.color)  # compute bonus based on rank progress
            if piece.color == stm:
                score += b  # passed pawn helps the side to move
            else:
                score -= b  # passed pawn helps the opponent (so it's bad for stm)

    # -----------------------------
    # 2) King danger near the king
    # -----------------------------
    # Nested helper function: counts how many "king-neighborhood squares"
    # are attacked by the attacker_color.
    def king_danger(king_sq: int, attacker_color: bool) -> int:
        danger = 0  # number of attacked squares around the king

        # chess.BB_KING_ATTACKS[king_sq] is a bitboard of squares a king could move to from king_sq.
        # SquareSet converts that bitboard to an iterable set of squares.
        for s in chess.SquareSet(chess.BB_KING_ATTACKS[king_sq]):
            # If attacker attacks that neighbor square, increment danger
            if board.is_attacked_by(attacker_color, s):
                danger += 1
        return danger  # higher means king is under more pressure in its vicinity

    myk = board.king(stm)        # our king square (side to move)
    oppk = board.king(not stm)   # opponent king square
    if myk is not None and oppk is not None:
        # Penalize if opponent attacks squares around our king
        score -= 15 * king_danger(myk, not stm)

        # Reward if we attack squares around opponent king
        score += 15 * king_danger(oppk, stm)

    # -----------------------------
    # 3) Being in check
    # -----------------------------
    # board.is_check() means the side to move is in check.
    # Since we evaluate from side-to-move perspective, being in check is bad.
    if board.is_check():
        score -= 30

    # -----------------------------
    # 4) Opening/middlegame king safety heuristic
    # -----------------------------
    # Add safety score for side to move's king...
    score += king_safety_score(board, stm)

    # ...and subtract safety score for opponent's king (since that's good for them, bad for us)
    score -= king_safety_score(board, not stm)

    return score  # final evaluation score from side-to-move perspective


# Search: TT + alpha-beta + quiescence
@dataclass
class TTEntry:
    depth: int                # search depth at which this TT entry was stored
    score: int                # evaluated score for this position (may be exact/bound)
    flag: int                 # 0 exact, 1 lower-bound, 2 upper-bound (alpha-beta storage scheme)
    best: chess.Move | None   # best move found from this position (for move ordering / PV)

TT_EXACT, TT_LOWER, TT_UPPER = 0, 1, 2  # named constants for readability
TT: dict[int, TTEntry] = {}             # transposition table: key -> TTEntry

# Killer move heuristic:
# For each ply, keep up to 2 moves that caused a beta-cutoff earlier at the same ply.
# These are often strong tactical "refutations" and help with move ordering.
KILLERS = [[None, None] for _ in range(128)]

# History heuristic:
# Typically tracks how often a move (from,to) caused cutoffs; used to order moves.
HISTORY = {}  # (from_square, to_square) -> score

def tt_key(board: chess.Board) -> int:
    # Use python-chess's internal Zobrist-like transposition key for hashing positions.
    # NOTE: this is a "private" method (leading underscore), but commonly used in engines.
    return board._transposition_key()


def mvv_lva_score(board: chess.Board, mv: chess.Move) -> int:
    # MVV-LVA = "Most Valuable Victim - Least Valuable Attacker"
    # Used for capture move ordering: prefer winning big pieces with small pieces.
    if not board.is_capture(mv):
        return 0  # non-captures get no MVV-LVA bonus

    victim = board.piece_at(mv.to_square)    # piece being captured (on destination square)
    attacker = board.piece_at(mv.from_square)  # capturing piece (from origin square)

    # Get victim material value (0 if None, though captures should usually have a victim unless en passant)
    v = PIECE_VALUES.get(victim.piece_type, 0) if victim else 0

    # Get attacker material value (0 if None; should exist in legal chess)
    a = PIECE_VALUES.get(attacker.piece_type, 0) if attacker else 0

    # Base 10,000 ensures captures beat quiet moves (in your ordering scoring scale)
    # Then add victim value, subtract a small fraction of attacker value to prefer cheaper attackers.
    return 10_000 + v - (a // 10)


def ordered_moves(board: chess.Board, tt_move: chess.Move | None, ply: int) -> list[chess.Move]:
    moves = list(board.legal_moves)  # generate all legal moves for current side to move

    def score(m: chess.Move) -> int:
        s = 0  # score accumulator for move ordering

        # If TT suggests a best move for this position, try it first (huge bonus)
        if tt_move is not None and m == tt_move:
            s += 2_000_000  # massive priority to improve alpha-beta cutoffs

        # Captures: prioritize using MVV-LVA heuristic
        if board.is_capture(m):
            s += mvv_lva_score(board, m)

        # Moves that give check: often tactical, so try early
        if board.gives_check(m):
            s += 500

        # Promotions are usually huge swings; try them early
        if m.promotion:
            s += 9_000

        # Strongly prefer castling early (improves king safety + rook activation)
        if board.is_castling(m):
            s += 2_000

        # Discourage early king moves that lose castling rights
        moved = board.piece_at(m.from_square)  # piece we are moving
        if moved and moved.piece_type == chess.KING and board.fullmove_number < 15 and not board.is_castling(m):
            s -= 3_000  # big penalty to avoid walking king around early

        # Extra opening heuristics (first 10 full moves):
        if board.fullmove_number < 10 and moved:
            if moved.piece_type == chess.QUEEN:
                s -= 1200  # discourage early queen development (often a tempo target)
            if moved.piece_type == chess.KING and not board.is_castling(m):
                s -= 3000  # (redundant with above, but enforces even more strongly)

        # Killer/history heuristics: only for quiet moves (non-captures)
        if not board.is_capture(m):
            k0, k1 = KILLERS[ply]  # retrieve killer moves stored for this ply depth

            # If move matches the first/second killer, boost it (likely causes cutoffs in similar positions)
            if k0 is not None and m == k0:
                s += 8_000
            elif k1 is not None and m == k1:
                s += 7_000

            # History heuristic: reward moves that historically caused beta cutoffs
            s += HISTORY.get((m.from_square, m.to_square), 0)

        return s  # final ordering score for this move

    # Sort all moves by score descending so we search "best-looking" moves first
    moves.sort(key=score, reverse=True)
    return moves  # return ordered move list


def quiescence(board: chess.Board, alpha: int, beta: int, stop_time: float) -> int:
    # Time control: if we're out of time, return a static evaluation immediately
    if time.perf_counter() >= stop_time:
        return evaluate(board)

    # "stand pat" evaluation: evaluate current position without making tactical moves
    stand_pat = evaluate(board)

    # If even standing still is already too good for the side to move, prune (fail-high)
    if stand_pat >= beta:
        return beta  # beta cutoff in quiescence

    # Otherwise, raise alpha if stand_pat improves it
    if alpha < stand_pat:
        alpha = stand_pat

    # Quiescence expansion: only explore tactical moves (captures + checks)
    for mv in board.legal_moves:
        # Skip quiet moves; we only want to resolve immediate tactics/noise
        if not board.is_capture(mv) and not board.gives_check(mv):
            continue

        # Optional pruning: skip captures you consider "obviously bad"
        # (this is your custom heuristic; be careful it doesn't miss tactical resources)
        if capture_is_obviously_bad(board, mv):
            continue

        board.push(mv)  # make the move
        # Negamax sign flip: opponent's perspective, so negate returned value
        score = -quiescence(board, -beta, -alpha, stop_time)
        board.pop()  # undo the move

        # Standard alpha-beta logic inside quiescence
        if score >= beta:
            return beta  # fail-high cutoff
        if score > alpha:
            alpha = score  # improve alpha with this tactical line

        # Time control inside loop: stop searching if time is up
        if time.perf_counter() >= stop_time:
            break

    return alpha  # best score found in quiescence window


def negamax(board: chess.Board, depth: int, alpha: int, beta: int, ply: int, stop_time: float) -> int:
    # Hard time cutoff: return static eval to avoid exceeding time budget
    if time.perf_counter() >= stop_time:
        return evaluate(board)

    # If game ended (mate/stalemate/etc), just return evaluation (your evaluate handles terminals)
    if board.is_game_over():
        return evaluate(board)

    key = tt_key(board)  # compute hash key for transposition table lookup
    entry = TT.get(key)  # see if we already evaluated this position
    tt_move = entry.best if entry else None  # best move stored for move ordering (if available)

    # Transposition Table (TT) cutoff logic:
    # If we have an entry searched at least as deep as current request, we can reuse it.
    if entry and entry.depth >= depth:
        if entry.flag == TT_EXACT:
            return entry.score  # exact score can be returned immediately
        if entry.flag == TT_LOWER:
            alpha = max(alpha, entry.score)  # lower bound raises alpha
        elif entry.flag == TT_UPPER:
            beta = min(beta, entry.score)   # upper bound lowers beta
        if alpha >= beta:
            return entry.score  # window collapsed -> cutoff based on TT bound

    # Depth reached: switch to quiescence to avoid horizon effect on tactics
    if depth == 0:
        return quiescence(board, alpha, beta, stop_time)

    best_score = -INFINITY  # track best score found so far
    best_move = None        # track best move that achieved best_score
    alpha0 = alpha          # remember original alpha to decide TT flag at the end

    # Iterate moves in "good" order to maximize alpha-beta pruning
    for mv in ordered_moves(board, tt_move, ply):

        # Optional pruning: skip moves that your heuristic says are tactical blunders
        if capture_is_obviously_bad(board, mv):
            continue

        board.push(mv)  # play candidate move

        # Recurse with negamax:
        # - flip signs and swap alpha/beta because perspective switches each ply
        score = -negamax(board, depth - 1, -beta, -alpha, ply + 1, stop_time)

        board.pop()  # undo move

        # Check time after returning from recursion (important in deep searches)
        if time.perf_counter() >= stop_time:
            break

        # Update best score/move if this move is better
        if score > best_score:
            best_score = score
            best_move = mv

        # Improve alpha if this move beats current alpha
        if score > alpha:
            alpha = score

        # Beta cutoff: opponent has a refutation, so no need to search remaining moves
        if alpha >= beta:
            # If this was a quiet move, store killer + history to improve future move ordering
            if not board.is_capture(mv):
                k0, k1 = KILLERS[ply]  # current killers for this ply

                # Update killer list: push current killer to second slot
                if k0 != mv:
                    KILLERS[ply][1] = k0
                    KILLERS[ply][0] = mv

                # Increase history score (deeper cutoffs = bigger reward)
                HISTORY[(mv.from_square, mv.to_square)] = HISTORY.get((mv.from_square, mv.to_square), 0) + depth * depth
            break  # stop searching other moves (cutoff)

    # -----------------------------
    # Store result in transposition table
    # -----------------------------
    flag = TT_EXACT  # assume exact unless proven bound

    # If best_score didn't beat the original alpha, it's an upper bound (fail-low)
    if best_score <= alpha0:
        flag = TT_UPPER

    # If best_score is >= beta, it's a lower bound (fail-high)
    elif best_score >= beta:
        flag = TT_LOWER

    # Save to TT: this lets future calls reuse the result and also gives a good tt_move for ordering
    TT[key] = TTEntry(depth=depth, score=best_score, flag=flag, best=best_move)

    return best_score  # final negamax score for this position


def search_best_move(board: chess.Board, max_depth: int, budget_sec: float) -> chess.Move:
    # Log an engine version string (useful for debugging which build is running on Lichess)
    logger.info("ENGINE VERSION: 2025-12-27 SAFEOPEN+SEE")

    # Clear the transposition table each move so old positions don't pollute this search
    TT.clear()

    # NOTE: you intentionally do NOT clear HISTORY/KILLERS so heuristics persist across moves (often helps)

    # Compute the absolute stop time for this search using perf_counter() (monotonic, good for timing)
    # Also enforce a minimum of 0.05 seconds so the engine has time to do something
    stop_time = time.perf_counter() + max(0.05, budget_sec)

    best_move = None           # best move found across completed iterative deepening depths
    best_score = -INFINITY     # score of best_move (from side-to-move perspective)

    # Iterative deepening: search depth=1, then 2, ... until max_depth or time runs out
    for depth in range(1, max_depth + 1):
        # Stop if we're out of time before starting the next depth
        if time.perf_counter() >= stop_time:
            break

        alpha, beta = -INFINITY, INFINITY  # reset aspiration window to full window at each iteration

        # Pull TT entry for this root position (if any) to use its stored best move for ordering
        entry = TT.get(tt_key(board))
        tt_move = entry.best if entry else None

        local_best = None              # best move found at THIS depth
        local_best_score = -INFINITY   # score for local_best at THIS depth

        # Loop through root moves in good order (TT move, captures, killers, etc.)
        for mv in ordered_moves(board, tt_move, ply=0):
            # Early pruning at root: in first ~40 plies, skip moves your heuristic says are obviously bad
            if board.ply() < 40 and capture_is_obviously_bad(board, mv):
                continue

            # --- FIX 2: root anti-sacrifice filter ---
            # In the early game, avoid "obvious" material sacrifices unless they are immediate mates.
            if board.ply() < 30:  # only apply this safety filter in the opening-ish phase
                delta = material_delta_if_play(board, mv)  # material swing from side-to-move perspective
                if delta < -150:  # losing more than ~1.5 pawns worth of material
                    # Allow the sacrifice only if it immediately checkmates.
                    board.push(mv)                 # make move
                    is_mate = board.is_checkmate() # checkmate means opponent is already mated
                    board.pop()                    # undo move
                    if not is_mate:
                        continue                   # reject the sacrifice
            # --- END FIX 2 ---

            board.push(mv)  # make the candidate root move (now opponent is to move)

            # NEW: don't allow moves that let opponent mate immediately (mate in 1 for the opponent)
            if opponent_has_mate_in_1(board):  # assumes opponent to move (true after push)
                board.pop()  # undo mv
                continue     # reject this root move

            # Immediate blunder filter: if opponent can capture our queen or rook right away, reject
            if hangs_major_piece_after(board):  # assumes opponent to move (true after push)
                board.pop()  # undo mv
                continue     # reject this root move

            # Queen trap avoidance: if our queen is attacked and has almost no mobility, reject (early game only)
            if board.ply() < 30 and queen_can_be_trapped_soon(board):  # also assumes opponent to move
                board.pop()  # undo mv
                continue     # reject this root move

            # Run the main negamax search on the resulting position:
            # - depth-1 because we already made 1 ply at root
            # - negate because negamax evaluates from side-to-move; after push it's opponent's turn
            score = -negamax(board, depth - 1, -beta, -alpha, 1, stop_time)

            board.pop()  # undo root move and restore original board state

            # If time is up after this move, stop evaluating more root moves at this depth
            if time.perf_counter() >= stop_time:
                break

            # Track best move for this depth
            if score > local_best_score:
                local_best_score = score
                local_best = mv

            # Improve alpha for root (this helps prune inside negamax on later root moves)
            if score > alpha:
                alpha = score

        # If we found any move at this depth, promote it to the global "best so far"
        if local_best is not None:
            best_move = local_best
            best_score = local_best_score

        # Log iterative deepening progress: depth, score, and best move so far
        logger.info("[ID depth=%d score=%d] best=%s", depth, best_score, best_move.uci() if best_move else "None")

    # If no move was found (should be rare), just play the first legal move to avoid crashing
    if best_move is None:
        return list(board.legal_moves)[0]

    return best_move  # final chosen move


def material_count(board: chess.Board, color: bool) -> int:
    total = 0  # accumulator for material value
    for ptype, val in PIECE_VALUES.items():
        # Count how many pieces of this type the side has, multiply by its value, add to total
        total += len(board.pieces(ptype, color)) * val
    return total  # total material for this color


def material_delta_if_play(board: chess.Board, mv: chess.Move) -> int:
    """
    Material change for the ORIGINAL side-to-move after playing mv.
    Negative => you sacrificed material.
    """
    stm = board.turn  # remember who is moving now (original side-to-move)

    # Compute material balance before move: (stm material) - (opponent material)
    before = material_count(board, stm) - material_count(board, not stm)

    board.push(mv)  # apply the move (board.turn flips after this)
    # After push, we still compute balance from the ORIGINAL stm perspective (not current board.turn)
    after = material_count(board, stm) - material_count(board, not stm)
    board.pop()  # undo move

    return after - before  # delta < 0 means stm lost material by playing mv



#Engine Wrapper
class ExampleEngine(MinimalEngine):
    """An example engine that all homemade engines inherit."""
    # This class exists mainly for organization / potential shared behavior.
    # It inherits from MinimalEngine (lichess-bot framework base class).

class SaidHybridEngine(ExampleEngine):
    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        # Read time controls and other search constraints passed by the lichess-bot framework
        time_limit = args[0] if len(args) > 0 and isinstance(args[0], Limit) else None  # timing info (may be None)

        root_moves = args[3] if len(args) > 3 else None  # optional list of allowed root moves (e.g., from book/filter)

        # Convert allowed root moves list to a set for fast membership testing
        allowed = set(root_moves) if isinstance(root_moves, list) and root_moves else None

        # 0) OPTIONAL: force a normal first move ONLY when we are White on move 1
        if board.ply() == 0 and board.turn == chess.WHITE:  # game start and we're White
            preferred = ["e2e4", "d2d4", "c2c4", "g1f3"]     # simple principled opening moves
            for uci in preferred:
                mv0 = chess.Move.from_uci(uci)              # build a move from UCI string
                # Ensure move is legal and also allowed (if an allowed list is enforced)
                if mv0 in board.legal_moves and (allowed is None or mv0 in allowed):
                    logger.info("[SAFE-OPEN-1] %s", mv0.uci())  # log the forced opening move
                    return PlayResult(mv0, None)                # return immediately without searching

        # 1) OPENING BOOK FIRST
        if board.ply() < BOOK_MAX_PLIES:          # only consult book in early opening phase
            mv = pick_book_move(board)            # attempt to pick a move from your opening book
            if mv is not None and (allowed is None or mv in allowed):  # ensure move exists and is allowed
                delta = material_delta_if_play(board, mv)              # safety: detect book sacrifices
                if delta < -100 and board.ply() < 20:                  # if it's a sacrifice early, consider skipping
                    logger.info("[BOOK-SKIP sacrifice delta=%d] %s", delta, mv.uci())
                else:
                    logger.info("[BOOK] %s", mv.uci())                 # accept book move
                    return PlayResult(mv, None)                        # play it immediately

        # 2) SAFE OPENING FALLBACK (BOTH COLORS)
        if board.ply() < BOOK_MAX_PLIES:          # still in early phase
            mv = safe_opening_move(board)         # try your hardcoded safe move list
            if mv is not None and (allowed is None or mv in allowed):  # ensure legal and allowed
                logger.info("[SAFE-OPEN] %s", mv.uci())
                return PlayResult(mv, None)       # play without heavy search

        # 3) TIME BUDGET
        budget = DEFAULT_BUDGET_SEC  # fallback budget if no time controls provided

        if time_limit is not None:
            # If time_limit.time is set (fixed time per move), use a fraction of it
            if isinstance(time_limit.time, (int, float)) and time_limit.time > 0:
                # clamp budget: at least 0.1s, at most 2.5s, and use ~60% of given per-move time
                budget = max(0.1, min(2.5, float(time_limit.time) * 0.6))
            else:
                # Otherwise use clock-based budgeting (typical lichess clock games)
                wc = time_limit.white_clock if isinstance(getattr(time_limit, "white_clock", None), (int, float)) else None
                bc = time_limit.black_clock if isinstance(getattr(time_limit, "black_clock", None), (int, float)) else None

                # Choose the correct clock depending on whose turn it is
                clock = wc if board.turn == chess.WHITE else bc

                if clock is not None:
                    # Use a simple "clock/80" allocation with clamping
                    budget = max(0.1, min(2.0, float(clock) / 80.0))

        # 4) SEARCH
        mv = search_best_move(board, max_depth=MAX_DEPTH, budget_sec=budget)  # run iterative deepening search

        # If the framework restricted allowed moves and our chosen move isn't allowed, fall back safely
        if allowed is not None and mv not in allowed:
            legal_allowed = [m for m in board.legal_moves if m in allowed]  # filter legal moves to allowed subset
            mv = legal_allowed[0] if legal_allowed else next(iter(board.legal_moves))  # pick something legal

        # Log the final move with time budget used
        logger.info("[MOVE t=%.2fs] %s", budget, mv.uci())

        return PlayResult(mv, None)  # return chosen move to lichess-bot framework


# extra engines (optional)
class RandomMove(ExampleEngine):
    def search(self, board: chess.Board, *args: HOMEMADE_ARGS_TYPE) -> PlayResult:
        # Pick a random legal move (for testing / baseline)
        return PlayResult(random.choice(list(board.legal_moves)), None)