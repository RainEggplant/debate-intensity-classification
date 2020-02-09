function Theta = LR_Newton(X, Y)
% ţ�ٷ��Ż�LR�㷨
% ������
% XΪ��������Ϊ200x13�ľ���
% Y�Ƕ�Ӧ�Ĵ�Ϊ200x1�ľ���
% ThetaΪѵ����Ĳ���

[n, m] = size(X);
Step = 100000;                       % ��������
J = zeros(1, Step);
Theta = zeros(m, 1)    % ��ʼ��Theta
while (true)
    G = Calc(X * Theta);            % �������ͨ�����������Ĵ�
    E = G - Y;                      % �������
    Delta = (1.0 / n) .* X' * E;     % ����һ���½�ֵ
    H = (1.0 / n) .*  X' * diag(G) * diag(1 - G) * X;
    Theta = Theta - H ^ (-1) * Delta;
    Step = Step - 1;        % ���Ʋ���
    if (Step <= 0)
        break;
    end
    HH = Calc(X * Theta);
    Cup = Y .* log(HH) + (1 - Y) .* log(1 - HH);
    J(100000-Step+1) = -mean(Cup);
end
plot(J);
