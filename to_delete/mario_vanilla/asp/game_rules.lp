{action(X): X = 0..6} = 1.

% you cannot perform NOOP, right, right B when enemy is just right of you.
:- action(1), cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C1 + 1 = C2.
:- action(3), cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C1 + 1 = C2.
:- action(0), cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C1 + 1 = C2.

% you cannot go left when enemy is just left of you.
:- action(6), cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C1 - 1 = C2.

% you cannot walk through a platform on the right of you.
:- action(1), cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.
:- action(3), cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.

% standing still against a platform achieves nothing.
:- action(0), cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.

% you cannot walk through a platform on the left of you.
:- action(6), cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 -1 1 = C2.

% You cannot run into a hole.


% You cannot jump into a hole.


% You cannot jump in front of an enemy.

