% mario = 1, platform = 2, enemy = 3, hole = 4, pipe = 5, wall = 6. 



enemy(XMIN, XMAX, YMIN, YMAX) :- goomba(XMIN,XMAX,YMIN,YMAX).
enemy(XMIN, XMAX, YMIN, YMAX) :- koopa_troopa(XMIN,XMAX,YMIN,YMAX).
enemy(XMIN, XMAX, YMIN, YMAX) :- piranha_plant(XMIN,XMAX,YMIN,YMAX).

platform(XMIN, XMAX, YMIN, YMAX) :- brick(XMIN,XMAX,YMIN,YMAX).
platform(XMIN, XMAX, YMIN, YMAX) :- ground(XMIN,XMAX,YMIN,YMAX).

%pipes have become their own category (5)
%platform(XMIN, XMAX, YMIN, YMAX) :- pipe(XMIN,XMAX,YMIN,YMAX).

platform(XMIN, XMAX, YMIN, YMAX) :- surprise_box(XMIN,XMAX,YMIN,YMAX).
platform(XMIN, XMAX, YMIN, YMAX) :- surprise_box_empty(XMIN,XMAX,YMIN,YMAX).

above(stone(XMIN1,XMAX1,YMIN1,YMAX1), stone(XMIN2,XMAX2,YMIN2,YMAX2)) :-  stone(XMIN1,XMAX1,YMIN1,YMAX1), stone(XMIN2,XMAX2,YMIN2,YMAX2),
                                                                          stone(XMIN1,XMAX1,YMIN1,YMAX1) != stone(XMIN2,XMAX2,YMIN2,YMAX2),
                                                                          CX = (XMIN1 + XMAX1) / 2, CY = (YMIN1 + YMAX1) / 2,
                                                                          CY2 = (YMIN2 + YMAX2) / 2,
                                                                          CX > XMIN2, CX < XMAX2,
                                                                          CY < CY2, CY > (YMIN2 - (YMAX1 - YMIN1)).

platform(X1,X2,Y1,Y2) :- stone(X1,X2,Y1,Y2), not above(_,stone(X1,X2,Y1,Y2)).

left(stone(XMIN1,XMAX1,YMIN1,YMAX1), stone(XMIN2,XMAX2,YMIN2,YMAX2)) :-  stone(XMIN1,XMAX1,YMIN1,YMAX1), stone(XMIN2,XMAX2,YMIN2,YMAX2),
                                                                          stone(XMIN1,XMAX1,YMIN1,YMAX1) != stone(XMIN2,XMAX2,YMIN2,YMAX2),
                                                                          CX = (XMIN1 + XMAX1) / 2, CY = (YMIN1 + YMAX1) / 2,
                                                                          CX2 = (XMIN2 + XMAX2) / 2,
                                                                          CY > YMIN2, CY < YMAX2,
                                                                          CX < CX2, CX > (XMIN2 - (XMAX1 - XMIN1)).

right(stone(XMIN1,XMAX1,YMIN1,YMAX1), stone(XMIN2,XMAX2,YMIN2,YMAX2)) :-  stone(XMIN1,XMAX1,YMIN1,YMAX1), stone(XMIN2,XMAX2,YMIN2,YMAX2),
                                                                          stone(XMIN1,XMAX1,YMIN1,YMAX1) != stone(XMIN2,XMAX2,YMIN2,YMAX2),
                                                                          CX = (XMIN1 + XMAX1) / 2, CY = (YMIN1 + YMAX1) / 2,
                                                                          CX2 = (XMIN2 + XMAX2) / 2,
                                                                          CY > YMIN2, CY < YMAX2,
                                                                          CX > CX2, CX < (XMAX2 + (XMAX1 - XMIN1)).

% a stone is considered a wall if there is a block above it and no block on either left or right hand side.
wall(X1,X2,Y1,Y2) :- stone(X1,X2,Y1,Y2), above(_,stone(X1,X2,Y1,Y2)), not left(_,stone(X1,X2,Y1,Y2)).
wall(X1,X2,Y1,Y2) :- stone(X1,X2,Y1,Y2), above(_,stone(X1,X2,Y1,Y2)), not right(_,stone(X1,X2,Y1,Y2)).

hole(X,255,YMIN,YMAX) :- ground(XMIN,X,YMIN,YMAX), X = #max{XMAX:ground(_,XMAX,_,_)}, X < 25.
hole(0,X,207,240) :- ground(X,XMAX,YMIN,YMAX), X = #min{XMIN:ground(XMIN,_,_,_)}, X > 5.

hole(XMAX1,XMIN2,YMIN1,YMAX1) :- adjacent(ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN2,XMAX2,YMIN2,YMAX2)).


-adjacent(ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN2,XMAX2,YMIN2,YMAX2)) :- ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN3,XMAX3,YMIN3,YMAX3), ground(XMIN2,XMAX2,YMIN2,YMAX2),
                                                                              ground(XMIN1,XMAX1,YMIN1,YMAX1) != ground(XMIN2,XMAX2,YMIN2,YMAX2), ground(XMIN1,XMAX1,YMIN1,YMAX1) != ground(XMIN3,XMAX3,YMIN3,YMAX3),
                                                                              ground(XMIN2,XMAX2,YMIN2,YMAX2) != ground(XMIN3,XMAX3,YMIN3,YMAX3), XMAX3 < XMIN2, XMIN3 > XMAX1.

adjacent(ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN2,XMAX2,YMIN2,YMAX2)) :- ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN2,XMAX2,YMIN2,YMAX2), ground(XMIN1,XMAX1,YMIN1,YMAX1) != ground(XMIN2,XMAX2,YMIN2,YMAX2), XMAX1 < XMIN2,
                                                                              not -adjacent(ground(XMIN1,XMAX1,YMIN1,YMAX1), ground(XMIN2,XMAX2,YMIN2,YMAX2)).


% convert to categorical cell representation
object(1, (XMIN + XMAX) / 2, (YMIN + YMAX) /2) :- mario(XMIN,XMAX,YMIN,YMAX).
object(3, (XMIN + XMAX) / 2, (YMIN + YMAX) /2) :- enemy(XMIN,XMAX,YMIN,YMAX).

%we do not need the bottom row as this only contains ground that can never be touched
center_cell(R,C, (C * 16) + 8, (R * 16) + 8) :- R = 0..13, C = 0..15.

% Fill the rest of the cells with nothing (i.e. 0). Unnecessary as Numpy does this for us.
% cell(R,C,0) :- R=0..14, C=0..15, not cell(R,C,1), not cell(R,C,2), not cell(R,C,3), not cell(R,C,4).

% place the game objects (moving agents) into a cell (16 x 16). (What if Mario becomes bigger by accident?)
% When center falls on cell border, X-wise they are snapped to the right, Y-wise they are snapped downwards
% can only be within 15*16 cell grid

cell(Y1, X1, C) :- object(C,X,Y), X1 = X / 16, X2 = X \ 16, X2 != 0,
                                    Y1 = Y / 16, Y2 = Y \ 16, Y2 != 0,
                                    Y1 < 15, Y1 >= 0, X1 >= 0, X1 < 16.

cell(Y1, X1 + 1, C) :- object(C,X,Y), X1 = X / 16, X2 = X \ 16, X2 = 0,
                                    Y1 = Y / 16, Y2 = Y \ 16, Y2 != 0,
                                    Y1 < 15, Y1 >= 0, X1 >= 0, X1 < 15.

cell(Y1 + 1, X1, C) :- object(C,X,Y), X1 = X / 16, X2 = X \ 16, X2 != 0,
                                    Y1 = Y / 16, Y2 = Y \ 16, Y2 = 0,
                                    Y1 < 14, Y1 >= 0, X1 >= 0, X1 < 15.

cell(Y1 + 1, X1 + 1, C) :- object(C,X,Y), X1 = X / 16, X2 = X \ 16, X2 = 0,
                                   Y1 = Y / 16, Y2 = Y \ 16, Y2 = 0,
                                   Y1 < 14, Y1 >= 0, X1 >= 0, X1 < 15.



% if the center of a cell is within a platform or a hole, then the cell is either a platform or a hole.
% if a center falls on the bounding box of platform, hole, wall or pipe, then the region is enlarged to capture the entire
% bounding box, holes are more dominant than platforms.

cell(R,C,2) :- platform(XMIN,XMAX,YMIN,YMAX), center_cell(R,C,XC,YC), XC >= XMIN, XC <= XMAX, YC <= YMAX, YC >= YMIN, not cell(R,C,4).


cell(R,C,4) :- hole(XMIN,XMAX,YMIN,YMAX), center_cell(R,C,XC,YC), XC >= XMIN, XC <= XMAX, YC <= YMAX, YC >= YMIN.

cell(R,C,5) :- pipe(XMIN,XMAX,YMIN,YMAX), center_cell(R,C,XC,YC), XC >= XMIN, XC <= XMAX, YC <= YMAX, YC >= YMIN.

cell(R,C,6) :- wall(XMIN,XMAX,YMIN,YMAX), center_cell(R,C,XC,YC), XC >= XMIN, XC <= XMAX, YC <= YMAX, YC >= YMIN.


%mario(110,126,167,183).
%ground(0,254,207,240).
%goomba(232,249,192,208).
%pipe(251,255,143,206).

%#show cell/3.