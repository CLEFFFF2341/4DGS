# Predeclared robustness checks after confirmation 01

Candidate and code remain frozen. These checks do not select its hyperparameters
or replace any failed confirmation case.

1. Stronger quadrature control: native Gauss8 for 6,400 updates on the three
   reserved clips at seed 3, from their exact shared initial states. It has eight
   times the training render count of the primary candidate/control. Compare its
   final common Gauss32 evaluation to the already fixed candidate Gauss1 model.
   This checks that the gain is not an artifact of a weak one-node control.
2. Longer training: development c120 and n120 seed0, both native Gauss1 and the
   same candidate, 30,000 updates from the old shared initial states. No checkpoint
   selection. This is still fixed capacity, not the full production Trainer.
   Report whether the benefit survives a longer horizon; it does not count as
   untouched final-set evidence.

If either check fails, preserve it and narrow the conclusion; do not silently
change the candidate or omit the check.
