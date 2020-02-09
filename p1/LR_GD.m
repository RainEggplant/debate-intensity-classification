function Theta = LR_GD(X, Y)
% �ݶ��½����Ż�LR�㷨
% ������
% XΪ��������Ϊ200x13�ľ���
% Y�Ƕ�Ӧ�Ĵ�Ϊ200x1�ľ���
% ThetaΪѵ����Ĳ���

[n, m] = size(X);           % n��������m������
Alpha = 0.001;              % ����/ѧϰ��
Step = 5000000;             % ��������
J = zeros(1, Step);
Theta = zeros(m, 1);        % ��ʼ��Theta
while true
    A = X * Theta;          % �������ͨ�����������Ĵ�
    E = Calc(A) - Y;        % �������
    Delta = Alpha .* X' * E;% �����½�ֵ
    Theta = Theta - Delta;  % �ݶ��½�
    Step = Step - 1;        % ���Ʋ���
    if (Step <= 0)
        break;
    end
    H = Calc(A);
    Cup = Y .* log(H) + (1 - Y) .* log(1 - H);
    J(5000000-Step+1) = -mean(Cup);
end
plot(J);
end
    