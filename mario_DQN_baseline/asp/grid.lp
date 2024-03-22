center_cell(R,C, (C * 16) + 8, (R * 16) + 8) :- R = 0..14, C = 0..15.

% Fill the rest of the cells with nothing (i.e. 0).
cell(R,C,0) :- R=0..14, C=0..15, not cell(R,C,1), not cell(R,C,2), not cell(R,C,3), not cell(R,C,4).

% place the game objects into a cell (16 x 16).
% When center falls on cell border, X-wise they are snapped to the right, Y-wise they are snapped downwards

cell(Y1, X1, C) :- object(C,X,Y), X1 = X / 16, X2 = X \ 16, X2 != 0,
                                    Y1 = Y / 16, Y2 = Y \ 16, Y2 != 0.

cell(Y1, X1 + 1, C) :- object(C,X,Y), X1 = X / 16, X2 = X \ 16, X2 = 0,
                                    Y1 = Y / 16, Y2 = Y \ 16, Y2 != 0.

cell(Y1 + 1, X1, C) :- object(C,X,Y), X1 = X / 16, X2 = X \ 16, X2 != 0,
                                    Y1 = Y / 16, Y2 = Y \ 16, Y2 = 0.

cell(Y1 + 1, X1 + 1, C) :- object(C,X,Y), X1 = X / 16, X2 = X \ 16, X2 = 0,
                                   Y1 = Y / 16, Y2 = Y \ 16, Y2 = 0.

% if the center of a cell is within a platform or a hole, then the cell is either a platform or a hole.
cell(R,C,2) :- platform(XMIN,XMAX,YMIN,YMAX), center_cell(R,C,XC,YC), XC > XMIN, XC < XMAX, YC < YMAX, YC > YMIN.
cell(R,C,4) :- hole(XMIN,XMAX,YMIN,YMAX), center_cell(R,C,XC,YC), XC > XMIN, XC < XMAX, YC < YMAX, YC > YMIN.


platform(18,66,17,34).


#show cell/3.
%#show center_cell/4.