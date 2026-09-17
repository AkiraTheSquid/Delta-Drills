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

    def test_aim_above_a_band_skips_it(self):
        # Seth 2026-09-11: "Significantly harder" x3 moved the aim to 79 and the
        # bottom-up walk still served the next band up. Bands more than REACH
        # below the aim are cleared; the walk starts at the first within reach.
        questions=[SimpleNamespace(id=i,difficulty_score=s) for i,s in self.scores.items()]
        self.assertEqual([q.id for q in next_band(questions,[],self.scores,target=61)],[3,4])
        self.assertEqual([q.id for q in next_band(questions,[],self.scores,target=79)],[5,6,7,8,9])

    def test_aim_above_every_band_serves_the_hardest_first(self):
        questions=[SimpleNamespace(id=i,difficulty_score=s) for i,s in self.scores.items()]
        self.assertEqual([q.id for q in next_band(questions,[],self.scores,target=95)],[5,6,7,8,9])
        # ...and walks DOWN once the hardest band is spent, not back to the bottom.
        fresh=[q for q in questions if q.difficulty_score<65]
        self.assertEqual([q.id for q in next_band(fresh,[],self.scores,target=95)],[3,4])

    def test_no_aim_keeps_the_bottom_up_walk(self):
        questions=[SimpleNamespace(id=i,difficulty_score=s) for i,s in self.scores.items()]
        self.assertEqual([q.id for q in next_band(questions,[],self.scores)],[1,2])
        self.assertEqual([q.id for q in next_band(questions,[],self.scores,target=30)],[1,2])

    def test_spent_reach_walks_down_one_band_at_a_time(self):
        # codex 2026-09-11: bands 45/55/65, aim 79 puts only 65 within reach; with
        # 65 spent the walk must hand back ONE band (the hardest left), never the
        # mixed remainder — a mixed list let the picker re-choose the drill on screen.
        questions=[SimpleNamespace(id=i,difficulty_score=s) for i,s in self.scores.items() if s<65]
        self.assertEqual([q.id for q in next_band(questions,[],self.scores,target=79)],[3,4])
        questions=[q for q in questions if q.difficulty_score<55]
        self.assertEqual([q.id for q in next_band(questions,[],self.scores,target=79)],[1,2])

    def test_always_one_band(self):
        questions=[SimpleNamespace(id=i,difficulty_score=s) for i,s in self.scores.items()]
        attempts=[attempt(i) for i in (1,2,3,4,5,6)]   # every band has its two
        out=next_band(questions,attempts,self.scores)
        self.assertEqual(len({q.difficulty_score for q in out}),1)

    def test_legacy_capacity_is_explicit_and_reachable(self):
        report=progress([attempt(1),attempt(2)],{1:45,2:65})
        self.assertTrue(report["ready"])
        self.assertEqual(report["coverage_shortfall"],4)


if __name__ == "__main__":
    unittest.main()
