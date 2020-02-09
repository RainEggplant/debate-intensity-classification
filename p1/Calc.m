function G = Calc(A)
% 计算矩阵A通过Sigmod函数后的结果
[n, m] = size(A);
G = zeros(n,m);
for i = 1 : n
    for j = 1 : m
        G(i,j) = 1 / (1 + exp(-A(i,j)));
    end
end
end
