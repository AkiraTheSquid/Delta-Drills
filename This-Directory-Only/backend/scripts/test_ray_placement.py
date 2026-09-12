"""Behavioral checks against the real graph and expanded exercise bank."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
from datetime import datetime, timezone, timedelta

os.environ["USER_DATA_DIR"] = tempfile.mkdtemp(prefix="ray-placement-")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import diagnostic as D, practice_targets as T, kc_graph as G
from app.adaptive import UserPracticeState, _save_user_state, _load_user_state
from app.prioritization import question_is_unlocked
from app.questions import get_question_by_id


class RayPlacementTest(unittest.TestCase):
    def start(self, name="ray"):
        s = UserPracticeState(user_id=name)
        D.start(s, scope=T.RAY)
        return s

    def answer(self, s, result):
        q = D.select_probe(s)
        self.assertIsNotNone(q)
        self.assertEqual(D.select_probe(s).id, q.id)
        self.assertIn(G._qmatrix()[q.id]["source"], ("kp-independent", "kp-integrated"))
        self.assertTrue(set(G.question_kcs(q.id)) <= set(D.assessed_kcs(s)))
        kc = s.diagnostic["pending"]["kc"]
        D.record_probe(s, q, result, elapsed_secs=20)
        return kc

    def test_scope_and_account_isolation(self):
        a, b = self.start(), UserPracticeState(user_id="friend")
        scope = set(D.assessed_kcs(a))
        self.assertTrue(set(T.ray_kcs()) <= scope)
        self.assertIn("numpy.linalg-basics", scope)
        self.assertNotIn("nn.modules", scope)
        self.assertEqual(b.practice_target, "all")
        self.assertFalse(b.diagnostic)
        for qid, row in G._qmatrix().items():
            if set(row["target_kcs"]) - scope:
                self.assertFalse(question_is_unlocked(a, get_question_by_id(qid)))

    def test_passes_jump_and_misses_check_prerequisites(self):
        s = self.start()
        self.assertEqual(self.answer(s, "correct"), "raytracing.segment-intersection")
        self.assertEqual(self.answer(s, "correct"), "raytracing.make-rays-2d")
        failed = self.answer(s, "incorrect")
        self.assertEqual(failed, "raytracing.mesh-visibility")
        parent = self.answer(s, "correct")
        self.assertIn(parent, G._registry()[failed]["prereqs"])
        while D.should_run(s):
            self.answer(s, "incorrect")
        self.assertLessEqual(len(s.diagnostic["probes"]), 8)
        self.assertFalse(s.atom_mastery)
        self.assertFalse(s.kc_ladder)
        self.assertFalse(s.kc_exposure)
        self.assertIn("numpy.constructors", T.readiness(s))
        self.assertNotIn(failed, T.readiness(s))
        self.assertEqual(G.kc_stage(s, "raytracing.segment-intersection"), "partial")
        self.assertTrue(set(G.frontier(s)) <= T.scope_kcs(T.RAY))
        self.assertNotIn("python.values-and-names", G.frontier(s))

    def test_novice_does_not_escalate_after_failure(self):
        s = self.start()
        self.answer(s, "dont_know")
        while D.should_run(s):
            self.answer(s, "dont_know")
        tested = {p["kc"] for p in s.diagnostic["probes"]}
        self.assertNotIn("raytracing.mesh-visibility", tested)
        self.assertNotIn("raytracing.triangle-intersection", tested)
        self.assertFalse(T.readiness(s))
        self.assertIn("python.values-and-names", G.frontier(s))

    def test_retake_persistence_and_revocation(self):
        s = self.start("persist")
        self.answer(s, "correct")
        D.finish(s)
        _save_user_state(s)
        loaded = _load_user_state(s.user_id)
        self.assertEqual(loaded.practice_target, T.RAY)
        self.assertEqual(T.readiness(loaded), T.readiness(s))
        kc = "numpy.constructors"
        loaded.kc_ladder[kc] = {"attempts": [{"correct": False, "stage": "partial",
            "ts": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat()}]}
        self.assertNotIn(kc, T.readiness(loaded))
        D.start(loaded, scope="all")
        self.assertGreater(len(D.assessed_kcs(loaded)), len(T.scope_kcs(T.RAY)))
        self.assertIn(T.RAY, loaded.practice_placements)
        loaded.practice_target = "all"
        self.assertFalse(T.readiness(loaded))

    def test_time_cap_and_spent_bank(self):
        s = self.start()
        for kc in D.assessed_kcs(s):
            for qid in G.questions_for_kc(kc):
                q = get_question_by_id(qid)
                s.get_subtopic_state(q.subtopic).served_question_ids.append(qid)
        self.assertIsNotNone(D.select_probe(s), "retakes must survive a spent practice bank")
        s.diagnostic["spent_secs"] = T.BUDGET_SECS
        self.assertTrue(D.should_finish(s))


if __name__ == "__main__":
    unittest.main()
