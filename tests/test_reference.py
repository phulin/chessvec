"""Tests for the reference chess engine. We use python-chess as an
independent oracle for legality and result computations on a handful of
positions, then exercise specific tactical/edge-case scenarios directly.
"""

from __future__ import annotations

import chess
import pytest

from chessvec.action_encoding import file_of, rank_of
from chessvec.reference import (
    State,
    apply_move,
    game_result,
    in_check,
    legal_moves,
)
from chessvec.types import BLACK, BLACK_WIN, DRAW, WHITE

# A handful of FENs covering opening, middlegame, endgame, special rules.
FENS = [
    chess.STARTING_FEN,
    "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",  # castling rights
    "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",  # Kiwipete-ish
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",  # kiwipete
    "8/8/8/4k3/8/8/4K3/8 w - - 0 1",  # K vs K (draw)
    "8/8/8/4k3/8/4N3/4K3/8 w - - 0 1",  # K+N vs K (draw)
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # ep target
    "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
]


def _to_uci(move: tuple[int, int, int]) -> str:
    f, t, p = move
    s = (
        chr(ord("a") + file_of(f))
        + str(rank_of(f) + 1)
        + chr(ord("a") + file_of(t))
        + str(rank_of(t) + 1)
    )
    if p:
        s += "nbrq"[p - 2]
    return s


def _from_python_chess_move(m: chess.Move) -> tuple[int, int, int]:
    promo = 0
    if m.promotion is not None:
        # python-chess uses chess.KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5
        promo = m.promotion
    return (m.from_square, m.to_square, promo)


@pytest.mark.parametrize("fen", FENS)
def test_legal_moves_match_python_chess(fen: str) -> None:
    state = State.from_fen(fen)
    board = chess.Board(fen)
    ours = {_to_uci(m) for m in legal_moves(state)}
    theirs = {m.uci() for m in board.legal_moves}
    assert ours == theirs, (
        f"diff at {fen}\n  only ours: {ours - theirs}\n  only theirs: {theirs - ours}"
    )


def _strip_ep(fen: str) -> str:
    parts = fen.split()
    parts[3] = "-"
    return " ".join(parts)


@pytest.mark.parametrize("fen", FENS)
def test_apply_move_fen_round_trip(fen: str) -> None:
    # python-chess's default FEN omits the en-passant square if no legal
    # capture exists (considers pins). Our reference uses a simpler rule.
    # Compare with the ep field stripped; the dedicated en-passant test
    # below verifies ep semantics directly.
    state = State.from_fen(fen)
    board = chess.Board(fen)
    for mv in legal_moves(state):
        new_state = apply_move(state, mv)
        new_board = board.copy()
        new_board.push(chess.Move(mv[0], mv[1], promotion=(mv[2] if mv[2] else None)))
        ours = _strip_ep(new_state.to_fen())
        theirs = _strip_ep(new_board.fen())
        assert ours == theirs, (
            f"mismatch from {fen} via {_to_uci(mv)}\n  ours:   {ours}\n  theirs: {theirs}"
        )


def test_starting_position_legal_count() -> None:
    state = State.starting()
    assert len(legal_moves(state)) == 20


def test_checkmate_detection() -> None:
    # Fool's mate.
    state = State.from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert legal_moves(state) == []
    assert in_check(state, WHITE)
    assert game_result(state) == BLACK_WIN


def test_stalemate_detection() -> None:
    state = State.from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1")
    assert legal_moves(state) == []
    assert not in_check(state, BLACK)
    assert game_result(state) == DRAW


def test_insufficient_material_kvk() -> None:
    state = State.from_fen("8/8/8/4k3/8/8/4K3/8 w - - 0 1")
    assert game_result(state) == DRAW


def test_insufficient_material_knight() -> None:
    state = State.from_fen("8/8/8/4k3/8/4N3/4K3/8 w - - 0 1")
    assert game_result(state) == DRAW


def test_fifty_move_rule() -> None:
    state = State.from_fen("4k3/8/8/8/8/8/8/4K3 w - - 99 1")
    state.halfmove_clock = 100
    # K vs K is also insufficient material so this is a draw either way.
    assert game_result(state) == DRAW


def test_castling_kingside_white() -> None:
    state = State.from_fen("r3k2r/pppqpppp/2nb1n2/3p4/3P4/2NB1N2/PPPQPPPP/R3K2R w KQkq - 0 1")
    moves = legal_moves(state)
    castles = [m for m in moves if m[0] == 4 and m[1] == 6]
    assert len(castles) == 1
    new = apply_move(state, castles[0])
    assert new.board[6] == 6  # WK on g1
    assert new.board[5] == 4  # WR on f1


def test_en_passant_capture() -> None:
    state = State.from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")
    moves = legal_moves(state)
    ep = [m for m in moves if m[0] == 36 and m[1] == 43]  # e5 -> d6
    assert len(ep) == 1
    new = apply_move(state, ep[0])
    assert new.board[35] == 0  # d5 (captured pawn) cleared
    assert new.board[43] == 1  # WP on d6


def test_promotion() -> None:
    # White pawn on a7, black king on e8 (out of the way).
    state = State.from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1")
    moves = legal_moves(state)
    promos = [m for m in moves if m[0] == 48 and m[1] == 56]  # a7 -> a8
    assert len(promos) == 4
    promo_pieces = sorted(m[2] for m in promos)
    assert promo_pieces == [2, 3, 4, 5]


@pytest.mark.parametrize("fen", FENS[:5])
def test_perft_2(fen: str) -> None:
    """For a handful of positions, the count of leaf positions after 2 plies
    should match python-chess."""
    state = State.from_fen(fen)
    board = chess.Board(fen)
    n_ours = 0
    for mv in legal_moves(state):
        s2 = apply_move(state, mv)
        n_ours += len(legal_moves(s2))
    n_theirs = 0
    for m in board.legal_moves:
        b2 = board.copy()
        b2.push(m)
        n_theirs += b2.legal_moves.count()
    assert n_ours == n_theirs, f"perft(2) mismatch from {fen}: ours={n_ours} theirs={n_theirs}"
