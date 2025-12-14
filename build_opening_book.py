import json
import time
from collections import defaultdict

import chess
import chess.pgn

# Optional: only needed if your file is .zst
try:
    import zstandard as zstd
except ImportError:
    zstd = None


def pgn_stream(path: str):
    """Return a text stream for PGN, supporting .pgn and .pgn.zst."""
    if path.endswith(".zst"):
        if zstd is None:
            raise RuntimeError("Install zstandard: pip install zstandard")
        fh = open(path, "rb")
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(fh)
        # Text wrapper so chess.pgn can read it
        import io
        return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def result_to_score(result: str) -> float:
    """Map PGN result to a numeric score from White's perspective."""
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return 0.0
    if result == "1/2-1/2":
        return 0.5
    return 0.5  # unknown -> neutral

def save_book(book_dict, meta, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "book": book_dict}, f, ensure_ascii=False)


def build_book(
    pgn_path: str,
    out_path: str = "opening_book.json",
    max_games: int = 50_000,
    max_plies: int = 16,   # 8 moves each side
    min_elo: int = 1600,   # filter stronger games if headers exist
):
    """
    Builds a simple opening book:
    fen -> {uci_move: {"n": count, "score_sum": sum_of_scores}}
    """
    book = defaultdict(lambda: defaultdict(lambda: {"n": 0, "score_sum": 0.0}))
    compact = {}  # fen -> move -> {n, avg}


    t0 = time.time()
    games = 0

    with pgn_stream(pgn_path) as pgn:
        while games < max_games:
            game = chess.pgn.read_game(pgn)
            if game is None:
                break

            headers = game.headers
            # Filter by Elo if present
            try:
                we = int(headers.get("WhiteElo", "0"))
                be = int(headers.get("BlackElo", "0"))
            except ValueError:
                we, be = 0, 0
            if min_elo and (we < min_elo or be < min_elo):
                continue

            result = headers.get("Result", "*")
            score_white = result_to_score(result)

            board = game.board()
            ply = 0
            for move in game.mainline_moves():
                if ply >= max_plies:
                    break

                fen = board.fen()  # full FEN (includes side to move etc.)
                uci = move.uci()

                # For Black to move, store score from the side-to-move perspective
                score_for_side_to_move = score_white if board.turn == chess.WHITE else (1.0 - score_white)

                entry = book[fen][uci]
                entry["n"] += 1
                entry["score_sum"] += score_for_side_to_move

                board.push(move)
                ply += 1

            games += 1
            if games % 5000 == 0:
                # update compact snapshot for writing
                for fen, moves in book.items():
                    cm = compact.setdefault(fen, {})
                    for uci, v in moves.items():
                        n = v["n"]
                        cm[uci] = {"n": n, "avg": v["score_sum"] / n}

                meta = {
                    "source": pgn_path,
                    "max_games": max_games,
                    "max_plies": max_plies,
                    "min_elo": min_elo,
                    "games_used": games,
                    "positions": len(compact),
                }
                save_book(compact, meta, out_path)
                print(f"Parsed {games} games... positions={len(compact)} (checkpoint saved to {out_path})")


    # Convert to a compact structure with average score
    # Final checkpoint/save
    for fen, moves in book.items():
        cm = compact.setdefault(fen, {})
        for uci, v in moves.items():
            n = v["n"]
            cm[uci] = {"n": n, "avg": v["score_sum"] / n}

    meta = {
        "source": pgn_path,
        "max_games": max_games,
        "max_plies": max_plies,
        "min_elo": min_elo,
        "games_used": games,
        "positions": len(compact),
    }
    save_book(compact, meta, out_path)
    print(f"Saved {out_path}. positions={len(compact)} games_used={games} in {time.time()-t0:.1f}s")



if __name__ == "__main__":
    # Example usage:
    # python build_opening_book.py data\lichess_db_standard_rated_2025-11.pgn.zst
    import sys
    if len(sys.argv) < 2:
        print("Usage: python build_opening_book.py <path_to_pgn_or_pgn.zst>")
        raise SystemExit(2)
    build_book(sys.argv[1])
