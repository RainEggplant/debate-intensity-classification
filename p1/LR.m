function y = LR(x)
Vars = load('LR_second_order.mat');
Theta = Vars.Theta;
Ans = Calc(x * Theta);
y = [];
for i = 1 : 100
    if (Ans(i) > 0.5)
        y = [y ; 1];
    else
        y = [y ; 0];
    end
end
end
