function Theta = LR_Newton(X, Y)
% 牛顿法优化LR算法
% 参数：
% X为输入特征为200x13的矩阵
% Y是对应的答案为200x1的矩阵
% Theta为训练后的参数

[n, m] = size(X);
Step = 100000;                       % 迭代次数
J = zeros(1, Step);
Theta = zeros(m, 1)    % 初始化Theta
while (true)
    G = Calc(X * Theta);            % 计算参数通过参数给出的答案
    E = G - Y;                      % 计算误差
    Delta = (1.0 / n) .* X' * E;     % 计算一阶下降值
    H = (1.0 / n) .*  X' * diag(G) * diag(1 - G) * X;
    Theta = Theta - H ^ (-1) * Delta;
    Step = Step - 1;        % 控制步数
    if (Step <= 0)
        break;
    end
    HH = Calc(X * Theta);
    Cup = Y .* log(HH) + (1 - Y) .* log(1 - HH);
    J(100000-Step+1) = -mean(Cup);
end
plot(J);
