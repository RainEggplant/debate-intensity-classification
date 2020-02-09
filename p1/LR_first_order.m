clear all; close all; clc;

X = [];
Y = [];

for i = 1 : 100
    filename = ['train\positive\', num2str(i - 1), '\feat.mat'];
    load(filename);
    X = [X ; feat];
    Y = [Y ; 1];
end

for i = 1 : 100
    filename = ['train\negative\', num2str(i - 1), '\feat.mat'];
    load(filename);
    X = [X ; feat];
    Y = [Y ; 0];
end

Theta = LR_GD(X, Y);

XX = [];
for i = 1 : 100
    filename = ['test\', num2str(i - 1), '\feat.mat'];
    load(filename);
    XX = [XX ; feat];
end

YY = [];

Ans = Calc(XX * Theta);
Error = 0;
for i = 1 : 100
    if (Ans(i) > 0.5)
        YY = [YY ; 1];
    else
        YY = [YY ; 0];
    end
end
