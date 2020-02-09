function Theta = LR_GD(X, Y)
% 梯度下降法优化LR算法
% 参数：
% X为输入特征为200x13的矩阵
% Y是对应的答案为200x1的矩阵
% Theta为训练后的参数

[n, m] = size(X);           % n个样本，m个特征
Alpha = 0.001;              % 步长/学习率
Step = 5000000;             % 迭代次数
J = zeros(1, Step);
Theta = zeros(m, 1);        % 初始化Theta
while true
    A = X * Theta;          % 计算参数通过参数给出的答案
    E = Calc(A) - Y;        % 计算误差
    Delta = Alpha .* X' * E;% 计算下降值
    Theta = Theta - Delta;  % 梯度下降
    Step = Step - 1;        % 控制步数
    if (Step <= 0)
        break;
    end
    H = Calc(A);
    Cup = Y .* log(H) + (1 - Y) .* log(1 - H);
    J(5000000-Step+1) = -mean(Cup);
end
plot(J);
end
    