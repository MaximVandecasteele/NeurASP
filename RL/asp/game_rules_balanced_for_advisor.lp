protected :- cell(R1, C1, 1), cell(R2, C1 + 1, 2), R2 > R1.

%danger zones enemies equals 3
jump :- cell(R1, C1, 1), cell(R2, C2, 3), R1 = R2, C2 <= C1 + 3, C2 > C1.

% You have to jump a the edge of a cliff.
jump :- cell(R1, C1, 1), cell(13, C1 + 1, 4), not protected.
%addition for problem 1
jump :- cell(R1, C1, 1), cell(13, C1, 4), not protected.

%addition for problem 2
jump :- cell(R1, C1, 1), cell(R1 + 1, C1 + 2, 4).
jump :- cell(R1, C1, 1), cell(R1 + 2, C1 + 2, 4).

%addition for problem 3
jump :- cell(9,7,1), cell(10,7,5), cell(12,11,3).
jump :- cell(9,7,1), cell(10,7,5), cell(12,10,3).
jump :- cell(9,7,1), cell(10,7,5), cell(12,12,3).

% jump when a platform is close to you. For a pipe you already jump from two cells before the pipe.
jump :- cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 + 1 = C2.
jump :- cell(R1, C1, 1), cell(R2, C2, 5), R1 = R2, C2 >= C1 + 1, C2 <= C1 + 2.
jump :- cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 + 1 = C2.

% you have to jump when a platform is left or right of you. (object detection fix)
jump :- cell(R1, C1, 1), cell(R2, C2, 6), R1 = R2, C1 - 1 = C2.
jump :- cell(R1, C1, 1), cell(R2, C2, 2), R1 = R2, C1 - 1 = C2.

% wait for piranha plant
%wait :- cell(R1, C1, 1), cell(R2, C2, 7), C1 >= C2 - 3, C1 < C2.

% wait for turtles
%wait :- cell(7,7,1), cell(8,7,2), cell(12,C,3), C > 10.

%don't jump againt goomba from above

%pyramid :- cell(R1,C1,2), cell(R2,C2,2), R2 = R1 -1, C2 = C1 + 1.
%wait :- pyramid, cell(R,_,3), R < 10.



action(2) :- jump.
action(0) :- wait.