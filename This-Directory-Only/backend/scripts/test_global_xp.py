#!/usr/bin/env python3
"""The global week board (app/global_xp.py), 2026-09-30.

Seth: the Learner Home leaderboard covers everyone, not one study group;
2026-10-01: everyone who has practised, named by default, renamable.
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
        self.assertEqual(rows[2]["display_name"], "b")  # the email's local part

    def test_names_shown_to_everyone(self):
        me = self.user("me@x.com", {MONDAY: 1.0})
        other = self.user("o@x.com", {MONDAY: 9.0})
        third = self.user("grace.hopper@x.com", {MONDAY: 5.0})
        self.join(other, self.group(other), "Their Group Name")
        rows = self.board(me)["rows"]
        self.assertEqual([r["display_name"] for r in rows], ["Their Group Name", "grace hopper", "me"])
        self.assertEqual(self.board(third)["week_end"], "2026-10-04")  # Monday through Sunday

    def test_chosen_name_wins_and_resets(self):
        me = self.user("me@x.com", {MONDAY: 1.0})
        self.join(me, self.group(me), "Group Me")
        self.assertEqual(global_xp.set_name(self.db, me, "  Speedy\t Gonzales "), "Speedy Gonzales")
        self.assertEqual(self.board(me)["rows"][0]["display_name"], "Speedy Gonzales")
        self.assertEqual(global_xp.set_name(self.db, me, "Again"), "Again")
        self.assertEqual(global_xp.set_name(self.db, me, ""), "Group Me")  # blank = the default
        self.assertEqual(self.board(me)["rows"][0]["display_name"], "Group Me")
        with self.assertRaises(ValueError):
            global_xp.set_name(self.db, me, "me@x.com")
        with self.assertRaises(ValueError):
            global_xp.set_name(self.db, me, "x" * (global_xp.NAME_MAX + 1))

    def test_last_week_does_not_count(self):
        me = self.user("me@x.com", {MONDAY - timedelta(days=1): 50.0})
        out = self.board(me)
        self.assertEqual(out["rows"], [{**out["rows"][0], "xp": 0.0}])

    def test_idle_learners_listed_at_zero_never_practised_left_off(self):
        me = self.user("me@x.com")  # no log at all — still listed, as you
        self.user("old@x.com", {MONDAY: 40.0}, log_age_days=9)  # untouched since last week: 0, not replayed
        self.user("zero@x.com", {})
        self.user("never@x.com")  # an account that never answered anything
        self.user("guest-1@guest.delta-drills.app", {})  # an idle guest session
        g = self.user("guest-2@guest.delta-drills.app", {MONDAY: 2.0}, log_age_days=1)
        (self.dir / "not-a-uuid.attempts.jsonl").write_text("{}\n")
        rows = self.board(me)["rows"]
        self.assertEqual([r["display_name"] for r in rows], [global_xp.GUEST, "me", "old", "zero"])
        self.assertTrue(rows[1]["is_you"])
        self.assertEqual([r["xp"] for r in rows], [2.0, 0.0, 0.0, 0.0])
        self.assertEqual([r["rank"] for r in rows], [1, 2, 2, 2])
        self.assertEqual(global_xp.set_name(self.db, g, "Named Guest"), "Named Guest")

    def test_test_accounts_left_off_unless_you(self):
        me = self.user("me@x.com", {MONDAY: 1.0})
        t = self.user("test-bob-f1aeba72@test.delta-drills.app", {MONDAY: 50.0})
        self.assertEqual([r["display_name"] for r in self.board(me)["rows"]], ["me"])
        rows = self.board(t)["rows"]  # signed in as the test account: still your own row
        self.assertEqual([(r["is_you"], r["xp"]) for r in rows], [(True, 50.0), (False, 1.0)])

    def test_no_identity_leaves(self):
        me = self.user("me@x.com", {MONDAY: 5.0})
        self.user("secret@x.com", {MONDAY: 7.0})
        text = repr(self.board(me))
        self.assertNotIn("@", text)
        self.assertNotIn(str(me.id), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
