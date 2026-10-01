#!/usr/bin/env python3
"""The global week board (app/global_xp.py), 2026-09-30.

Seth: the Learner Home leaderboard covers everyone, not one study group.
Synthetic attempt-log files (only their mtime matters) and an isolated
SQLite DB; `learning_xp.replay` is stubbed with fixed per-day XP.

Run: .venv/bin/python scripts/test_global_xp.py
"""
import os
import sys
import tempfile
import unittest
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("USER_DATA_DIR", tempfile.mkdtemp(prefix="global_xp_test_"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import Session  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

from app import global_xp, learning_xp  # noqa: E402
from app.db import Base  # noqa: E402
from app.models import StudyGroup, StudyGroupMember, User  # noqa: E402

NOW = datetime(2026, 9, 30, 15, tzinfo=timezone.utc)  # a Wednesday
MONDAY = date(2026, 9, 28)


class BoardTests(unittest.TestCase):
    def setUp(self):
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(engine)
        self.db = Session(engine)
        self.dir = Path(tempfile.mkdtemp(prefix="global_xp_logs_"))
        self.xp = {}  # user id -> {date: xp}

    def group(self, owner):
        g = StudyGroup(id=uuid.uuid4(), name="g", owner_user_id=owner.id, join_token=uuid.uuid4().hex)
        self.db.add(g)
        self.db.commit()
        return g

    def join(self, u, g, name):
        self.db.add(StudyGroupMember(id=uuid.uuid4(), group_id=g.id, user_id=u.id, display_name=name))
        self.db.commit()

    def user(self, email, week_days=None, log_age_days=0.0):
        u = User(id=uuid.uuid4(), email=email, password_hash="x")
        self.db.add(u)
        self.db.commit()
        if week_days is not None:
            p = self.dir / f"{u.id}.attempts.jsonl"
            p.write_text("{}\n")
            t = (NOW - timedelta(days=log_age_days)).timestamp()
            os.utime(p, (t, t))
            self.xp[str(u.id)] = week_days
        return u

    def board(self, me):
        days = [MONDAY + timedelta(days=i) for i in range(-3, 3)]

        def replay(state, zone, now):
            row = self.xp.get(str(state.user_id), {})
            return {"days": days, "xp": [row.get(d, 0.0) for d in days]}

        with patch.object(learning_xp, "replay", side_effect=replay):
            return global_xp.read_board(self.db, me, timezone.utc, now=NOW, data_dir=self.dir)

    def test_everyone_active_this_week_ranked(self):
        me = self.user("me@x.com", {MONDAY: 10.0})
        a = self.user("a@x.com", {MONDAY: 30.0, MONDAY + timedelta(days=1): 5.0})
        self.user("b@x.com", {MONDAY: 10.0})
        g = self.group(a)
        self.join(a, g, "Ada")
        self.join(me, g, "Me")
        out = self.board(me)
        self.assertEqual(out["week_start"], "2026-09-28")
        rows = out["rows"]
        self.assertEqual([r["xp"] for r in rows], [35.0, 10.0, 10.0])
        self.assertEqual([r["rank"] for r in rows], [1, 2, 2])  # ties share a place
        self.assertTrue(rows[1]["is_you"])  # a tie puts you first among equals
        self.assertEqual(rows[1]["display_name"], "Me")
        self.assertEqual(rows[0]["display_name"], "Ada")
        self.assertEqual(rows[2]["display_name"], global_xp.ANON)

    def test_name_only_shown_to_a_groupmate(self):
        me = self.user("me@x.com", {MONDAY: 1.0})
        other = self.user("o@x.com", {MONDAY: 9.0})
        theirs, mine = self.group(other), self.group(me)
        self.join(other, theirs, "Private Name")
        self.join(me, mine, "Me")
        rows = self.board(me)["rows"]
        self.assertEqual([r["display_name"] for r in rows], [global_xp.ANON, "Me"])

    def test_last_week_does_not_count(self):
        me = self.user("me@x.com", {MONDAY - timedelta(days=1): 50.0})
        out = self.board(me)
        self.assertEqual(out["rows"], [{**out["rows"][0], "xp": 0.0}])

    def test_idle_and_zero_learners_left_off_you_kept(self):
        me = self.user("me@x.com")  # no log at all
        self.user("old@x.com", {MONDAY: 40.0}, log_age_days=9)  # log untouched since last week
        self.user("zero@x.com", {})
        (self.dir / "not-a-uuid.attempts.jsonl").write_text("{}\n")
        rows = self.board(me)["rows"]
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["is_you"])
        self.assertEqual(rows[0]["xp"], 0.0)

    def test_no_identity_leaves(self):
        me = self.user("me@x.com", {MONDAY: 5.0})
        self.user("secret@x.com", {MONDAY: 7.0})
        text = repr(self.board(me))
        self.assertNotIn("@", text)
        self.assertNotIn(str(me.id), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
