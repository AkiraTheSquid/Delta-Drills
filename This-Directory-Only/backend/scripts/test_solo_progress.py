"""Regression checks for progression, not just authored question counts."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.solo_progress import next_band, progress


def attempt(qid, correct=True, stage="partial", example=False):
    return dict(question_id=qid, correct=correct, stage=stage, example=example)


class SoloProgressTests(unittest.TestCase):
    scores = {1:45, 2:45, 3:55, 4:55, 5:65, 6:65, 7:65, 8:65, 9:65}

    def test_easy_streak_cannot_skip_harder_work(self):
        self.assertFalse(progress([attempt(i) for i in (1,2,3)], self.scores)["ready"])
        self.assertFalse(progress([attempt(i) for i in (1,2,3,4,5)], self.scores)["ready"])
        self.assertTrue(progress([attempt(i) for i in (1,2,3,4,5,6)], self.scores)["ready"])

    def test_duplicates_aided_and_other_rungs_do_not_buy_solo_credit(self):
        attempts = [attempt(1)]*8 + [attempt(2, example=True), attempt(3, stage="faded")]
        report = progress(attempts, self.scores)
        self.assertEqual(report["correct"], 1)
        self.assertFalse(report["ready"])

    def test_latest_failure_replaces_old_success(self):
        self.assertEqual(progress([attempt(1), attempt(1, False)], self.scores)["correct"],0)

    def test_actual_sequence_reaches_hard_band_without_repeating(self):
        questions=[SimpleNamespace(id=i,difficulty_score=s) for i,s in self.scores.items()]
        attempts=[]
        order=[]
        while not progress(attempts,self.scores)["ready"]:
            fresh=[q for q in questions if q.id not in order]
            chosen=next_band(fresh,attempts,self.scores)[0]
            order.append(chosen.id)
            attempts.append(attempt(chosen.id))
            self.assertLessEqual(len(order),len(questions))
        self.assertEqual([self.scores[i] for i in order],[45,45,55,55,65,65])

    def test_miss_does_not_trap_learner_in_empty_band(self):
        attempts=[attempt(1,False),attempt(2)]
        fresh=[SimpleNamespace(id=i,difficulty_score=s) for i,s in self.scores.items() if i>2]
        self.assertEqual([q.id for q in next_band(fresh,attempts,self.scores)],[3,4])

    def test_legacy_capacity_is_explicit_and_reachable(self):
        report=progress([attempt(1),attempt(2)],{1:45,2:65})
        self.assertTrue(report["ready"])
        self.assertEqual(report["coverage_shortfall"],4)


if __name__ == "__main__":
    unittest.main()
