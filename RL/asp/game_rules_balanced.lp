{action(X): X = 0..4} = 1.

protected :- cell(R1, C1, 1), cell(R2, C1 + 1, 2), R2 > R1.
jump :- cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C2 <= C1 + 2, C2 > C1.
jump :- cell(R1, C1, 1), cell(13, C1 + 1, 4), not protected.
jump :- cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.
jump :- cell(R1, C1, 1), cell(R2, C2, 5), R1 = R2, C2 >= C1 + 1, C2 <= C1 + 2.
jump :- cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 + 1 = C2.
jump :- cell(R1, C1, 1), not cell(R1 + 1, C1, _).

:- action(1), jump.

:- action(2), not jump.

:- action(0).
:- action(3).
:- action(4).
