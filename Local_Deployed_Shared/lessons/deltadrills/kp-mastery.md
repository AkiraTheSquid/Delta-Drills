---
kc: deltadrills.mastery
kind: math
title: What "mastery" means in this app
supporting: []
new_syntax: []
previews: []
concepts: []
faded: []
guided: []
independent: [50112, 50113]
integrated: []
---

## Concept

Every concept you practice has a mastery estimate: a number between 0 and 1,
the app's belief that you know the concept. It is not a percent of questions
answered correctly — it is a probability, and it moves every time you answer.

The update reads only two things: where the estimate sits now, and whether
this answer was right. The estimate already sums up your whole history on the
concept, so two concepts at the same number move the same way on the same
answer, however many answers got them there. What sets the size of the move
is the number itself. The model expects someone who knows a concept to slip
about one time in ten, and someone who doesn't to guess right about one time
in five — so a miss from 0.99 barely dents the estimate, while a miss from 0.9
cuts it roughly in half. The app calls a concept learned once its estimate
reaches 0.85: that is what unlocks the next concept in the graph and, on
ARENA concepts, the real exercise.

## Worked example

Say a concept's estimate sits at 0.55. You answer a question on it
correctly. A lucky guess is fairly rare, so one correct answer is strong
evidence: the estimate jumps to about 0.89 — past 0.85, so the concept is
marked learned and the picker turns to what comes after it in the graph.

Now take a concept at 0.99 after a long correct streak, and miss one
question. A slip by someone who knows the concept is expected now and then,
so the estimate falls only to about 0.93 — still learned. The same miss from
0.9 lands near 0.53: at 0.9 the model still had real doubt, and a miss is
exactly the evidence that doubt predicted.

## Solo practice

### q50112
A concept's mastery estimate is 0.99 after a long streak of correct answers.
You answer the next question wrong. What is the most likely effect?

### q50113
Two concepts both sit at mastery 0.6. One has ten prior answers behind that
number; the other has one. You get the next question wrong on both. Which
estimate drops further?
