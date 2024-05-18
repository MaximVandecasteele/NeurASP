protected :- cell(R1, C1, 1), cell(R2, C1 + 1, 2), R2 > R1.

%Je springt vanaf 3 posities voor een enemy
jump :- cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C2 <= C1 + 3, C2 > C1.

%Je springt wanneer je aan de edge van een gat staat
jump :- cell(R1, C1, 1), cell(13, C1 + 1, 4), not protected.
%addition for problem 3
jump :- cell(R1, C1, 1), cell(13, C1, 4), not protected.


%Bij een gat spring je vanaf twee posities voor het gat
jump :- cell(R1, C1, 1), cell(R1 + 1, C1 + 2, 4).
%addition for problem 2
jump :- cell(R1, C1, 1), cell(R1 + 2, C1 + 2, 4).

%exaggeration
jump :- cell(9,7,1), cell(10,7,5), cell(12,11,3).
jump :- cell(9,7,1), cell(10,7,5), cell(12,10,3).
jump :- cell(9,7,1), cell(10,7,5), cell(12,12,3).


% jump als platform, pipe, wall naast je staat. Bij een pipe spring je vanaf twee posities voor het gat
jump :- cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.
jump :- cell(R1, C1, 1), cell(R2, C2, 5), R1 = R2, C2 >= C1 + 1, C2 <= C1 + 2.
jump :- cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 + 1 = C2.

% je moet springen als er een platform of een wall links van je staat (dit is een fix voor de object detection)
jump :- cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 - 1 = C2.
jump :- cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 - 1 = C2.

% on ground
action(8) :- cell(R1, C1, 1), cell(R1 + 1, C1, 2).
action(8) :- cell(R1, C1, 1), cell(R1 + 1, C1, 5).
action(8) :- cell(R1, C1, 1), cell(R1 + 1, C1, 6).


action(2) :- jump.