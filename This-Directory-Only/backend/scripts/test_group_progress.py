"""Group chart regressions. Synthetic logs and an isolated SQLite DB only."""
import sys
import unittest
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app import attempt_log, kc_graph, study_group_progress as progress
from app.auth import get_current_user
from app.db import Base, get_db
from app.models import User, StudyGroup, StudyGroupMember, StudyGroupTarget
from app.practice.groups_router import router
from app.practice.activity_router import range_counts


class HistoryTests(unittest.TestCase):
    def test_canonical_mastery_uses_historical_clock(self):
        state = SimpleNamespace(atom_mastery={"atom-a": .8},
                                atom_last_ts={"atom-a": "2024-01-01T00:00:00Z"}, self_reported_level=None)
        mapping = {"kc-a": {"atoms": [{"a": "atom-a", "w": 1}], "tier": "measured"}}
        with patch.object(kc_graph, "_crosswalk", return_value=mapping):
            value, coverage, tier = kc_graph.kc_mastery(state, "kc-a", now=datetime(2024, 1, 1, tzinfo=timezone.utc))
        self.assertAlmostEqual(value, .8)
        self.assertEqual((coverage, tier), (1, "measured"))

    def test_activity_dst_preserves_local_calendar_days(self):
        with tempfile.TemporaryDirectory(prefix="group-progress-") as tmp:
            root = Path(tmp)
            for stamp in ["2026-03-08T05:30:00Z", "2026-03-09T04:30:00Z"]:
                attempt_log.append(attempt_log.AttemptRow(ts=stamp, kind="attempt", user_id="test", correct=True), root)
            result = range_counts("test", date(2026, 3, 7), date(2026, 3, 9), base_dir=root,
                                  tz_name="America/Chicago", now=datetime(2026, 3, 10, tzinfo=timezone.utc))
        self.assertEqual([day["count"] for day in result["days"]], [1, 1, 0])

    def test_calendar_boundaries(self):
        self.assertEqual(progress.calendar_range(date(2024, 2, 29), "monthly"),
                         (date(2024, 2, 1), date(2024, 2, 29)))
        self.assertEqual(progress.calendar_range(date(2026, 1, 1), "weekly"),
                         (date(2025, 12, 29), date(2026, 1, 4)))

    def test_history_denominator_proxies_gaps_and_future(self):
        areas = [{"id": "0.0", "kcs": ["a", "b"]}, {"id": "0.1", "kcs": ["c"]}]
        state = SimpleNamespace(atom_mastery={}, atom_last_ts={}, self_reported_level=None)
        def reading(past, kc, now=None):
            return past.atom_mastery.get(kc, .2), float(kc in past.atom_mastery), "measured" if kc == "a" else "topic-proxy"
        row = attempt_log.AttemptRow(ts="2026-09-02T12:00:00Z", kind="bkt_update", user_id="test",
                                    feature_sources={"bkt_changed": {"a": {"a": .8}}})
        row2 = attempt_log.AttemptRow(ts="2026-09-03T12:00:00Z", kind="bkt_update", user_id="test",
                                     feature_sources={"bkt_changed": {"b": {"b": .8, "c": .8}}})
        with patch.object(kc_graph, "kc_mastery", side_effect=reading):
            days = progress.mastery_history(state, [row, row2], areas, date(2026, 9, 1), date(2026, 9, 5),
                                            timezone.utc, datetime(2026, 9, 4, 12, tzinfo=timezone.utc))
        self.assertIsNone(days[0]["scores"]["aggregate"])
        self.assertEqual(days[1]["scores"]["0.0"]["score"], 50)  # .8 + unseen .2, divided by both concepts
        self.assertEqual(days[1]["scores"]["aggregate"]["score"], 40)  # weighted by 3 concepts, not 2 sections
        self.assertEqual(days[1]["scores"]["0.0"]["coverage"], 50)
        self.assertEqual(days[2]["scores"]["aggregate"]["score"], 80)
        self.assertEqual(days[2]["scores"]["aggregate"]["proxies"], 2)
        self.assertIsNone(days[4]["scores"]["aggregate"])

    def test_snapshot_does_not_backfill_old_days(self):
        state = SimpleNamespace(atom_mastery={"a": .9}, atom_last_ts={"a": "2026-09-03T12:00:00Z"}, self_reported_level=None)
        with patch.object(kc_graph, "kc_mastery", side_effect=lambda s, k, now: (s.atom_mastery.get(k, .2), float(k in s.atom_mastery), "measured")):
            days = progress.mastery_history(state, [], [{"id": "0.0", "kcs": ["a"]}], date(2026, 9, 1), date(2026, 9, 4),
                                            timezone.utc, datetime(2026, 9, 4, 12, tzinfo=timezone.utc))
        self.assertIsNone(days[1]["scores"]["0.0"])
        self.assertEqual(days[2]["scores"]["0.0"]["score"], 90)

    def test_dst_uses_calendar_zone_for_history(self):
        # 04:30 UTC is still March 8 in Chicago, after DST has changed.
        state = SimpleNamespace(atom_mastery={"a": .8}, atom_last_ts={"a": "2026-03-09T04:30:00Z"}, self_reported_level=None)
        with patch.object(kc_graph, "kc_mastery", side_effect=lambda s, k, now: (s.atom_mastery.get(k, .2), float(k in s.atom_mastery), "measured")):
            days = progress.mastery_history(state, [], [{"id": "0.0", "kcs": ["a"]}], date(2026, 3, 7), date(2026, 3, 9),
                                            ZoneInfo("America/Chicago"), datetime(2026, 3, 10, tzinfo=timezone.utc))
        self.assertIsNone(days[0]["scores"]["0.0"])
        self.assertEqual(days[1]["scores"]["0.0"]["score"], 80)


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        Base.metadata.create_all(self.engine)
        self.db = Session(self.engine)
        self.users = [User(email=f"test{i}@example.test", password_hash="unused") for i in range(3)]
        self.db.add_all(self.users)
        self.db.flush()
        self.groups = [StudyGroup(name=f"Group {i}", join_token=str(i) * 32, owner_user_id=self.users[i].id) for i in (0, 2)]
        self.db.add_all(self.groups)
        self.db.flush()
        self.members = [StudyGroupMember(user_id=user.id, group_id=self.groups[0 if i < 2 else 1].id, display_name=f"Learner {i}")
                        for i, user in enumerate(self.users)]
        self.db.add_all(self.members)
        self.db.commit()
        app = FastAPI()
        app.include_router(router)
        app.dependency_overrides[get_current_user] = lambda: self.users[0]
        app.dependency_overrides[get_db] = lambda: self.db
        self.app = app
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.db.close()
        self.engine.dispose()

    def test_target_persistence_ownership_validation_and_leave(self):
        good = {"area": "aggregate", "date": "2026-10-12", "level": 80}
        self.assertEqual(self.client.put("/groups/target", json=good).status_code, 200)
        self.assertEqual(self.client.put("/groups/target", json={**good, "level": 85}).status_code, 200)
        rows = self.db.query(StudyGroupTarget).all()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0].user_id, rows[0].level), (self.users[0].id, 85))
        for bad in ({"level": 101}, {"level": -1}, {"level": 80.5}, {"member_id": str(self.members[1].id)}):
            self.assertEqual(self.client.put("/groups/target", json={**good, **bad}).status_code, 422)
        self.assertEqual(self.client.put("/groups/target", json={**good, "area": "all"}).status_code, 400)
        self.assertEqual(self.client.put("/groups/target", json={**good, "date": "not-a-day"}).status_code, 400)
        self.db.delete(self.members[0])
        self.db.commit()
        self.assertEqual(self.client.put("/groups/target", json=good).status_code, 404)
        self.assertEqual(self.client.get("/groups/progress?date=2026-09-13").status_code, 404)

    def test_progress_only_contains_current_group_and_member_ids(self):
        state = SimpleNamespace(atom_mastery={}, atom_last_ts={}, self_reported_level=None)
        with patch.object(progress, "get_user_state", return_value=state), \
             patch.object(attempt_log, "iter_rows", return_value=[]), \
             patch.object(progress, "range_counts", return_value={"days": [], "total": 0}):
            response = self.client.get("/groups/progress?date=2024-02-29&horizon=monthly&tz_name=America/Chicago")
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["end"], "2024-02-29")
        self.assertEqual(set(body["entries"]), {str(m.id) for m in self.members[:2]})
        for user in self.users:
            self.assertNotIn(str(user.id), response.text)
            self.assertNotIn(user.email, response.text)
        self.assertEqual(len(next(iter(body["entries"].values()))["history"]), 29)

    def test_invalid_ranges_and_missing_auth(self):
        for query in ("horizon=yearly", "tz_name=No/SuchZone", "date=bad", "tz_offset=2000"):
            res = self.client.get("/groups/progress?date=2026-09-13&" + query)
            self.assertIn(res.status_code, (400, 422))
        del self.app.dependency_overrides[get_current_user]
        self.assertIn(self.client.get("/groups/progress?date=2026-09-13").status_code, (401, 403))


if __name__ == "__main__":
    unittest.main(verbosity=2)
