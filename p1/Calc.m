function G = Calc(A)
% �������Aͨ��Sigmod������Ľ��
[n, m] = size(A);
G = zeros(n,m);
for i = 1 : n
    for j = 1 : m
        G(i,j) = 1 / (1 + exp(-A(i,j)));
    end
end
end
