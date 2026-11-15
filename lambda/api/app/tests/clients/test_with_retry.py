"""DSQL の OCC リトライ付きトランザクション（with_retry）のテスト。

想定している正しい挙動:
- 再利用する接続にトランザクションが開いたまま残っていても（死活確認の SELECT 1 や前回の呼び出しの後）、
  先に閉じてから autocommit を切り替えて処理する。psycopg2 はトランザクション中の autocommit 変更を拒否する。
- 処理が成功したら commit して結果を返す。
- SerializationFailure は回数上限まで再試行し、それ以外の例外は rollback して再送出する。
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import psycopg2
import pytest

import clients.dsql as dsql


class _FakeConn:
    """トランザクション中の autocommit 変更を拒否する psycopg2 接続の代役。"""

    def __init__(self, in_transaction=False):
        self.status = (
            psycopg2.extensions.STATUS_IN_TRANSACTION if in_transaction else psycopg2.extensions.STATUS_READY
        )
        self._autocommit = False
        self.calls = []

    @property
    def autocommit(self):
        return self._autocommit

    @autocommit.setter
    def autocommit(self, value):
        if self.status != psycopg2.extensions.STATUS_READY:
            raise psycopg2.ProgrammingError("set_session cannot be used inside a transaction")
        self._autocommit = value

    def commit(self):
        self.calls.append("commit")
        self.status = psycopg2.extensions.STATUS_READY

    def rollback(self):
        self.calls.append("rollback")
        self.status = psycopg2.extensions.STATUS_READY


@pytest.fixture
def conn(monkeypatch):
    holder = {"conn": _FakeConn()}
    monkeypatch.setattr(dsql, "get_connection", lambda: holder["conn"])
    monkeypatch.setattr(dsql.time, "sleep", lambda _s: None)
    return holder


def test_closes_open_transaction_before_switching_autocommit(conn):
    conn["conn"] = _FakeConn(in_transaction=True)

    assert dsql.with_retry(lambda c: "ok") == "ok"
    assert conn["conn"].calls == ["rollback", "commit"]


def test_consecutive_calls_on_reused_connection_succeed(conn):
    def work(c):
        # 処理中はトランザクションが開いている。commit 後の再利用時に死活確認でまた開く想定
        c.status = psycopg2.extensions.STATUS_IN_TRANSACTION
        return "done"

    assert dsql.with_retry(work) == "done"
    conn["conn"].status = psycopg2.extensions.STATUS_IN_TRANSACTION  # get_connection の SELECT 1
    assert dsql.with_retry(work) == "done"


def test_retries_serialization_failure(conn):
    attempts = {"n": 0}

    def flaky(c):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise psycopg2.errors.SerializationFailure()
        return "ok"

    assert dsql.with_retry(flaky) == "ok"
    assert attempts["n"] == 2


def test_other_errors_roll_back_and_propagate(conn):
    def boom(c):
        c.status = psycopg2.extensions.STATUS_IN_TRANSACTION
        raise ValueError("boom")

    with pytest.raises(ValueError):
        dsql.with_retry(boom)
    assert conn["conn"].calls[-1] == "rollback"
    assert conn["conn"].status == psycopg2.extensions.STATUS_READY
