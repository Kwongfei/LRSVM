%{
README.txt:
[w,b,e] = LRSVM_1_SVM_para(A,B,Alpha,X,y)
��֪A,B,Alpha�����LRSVM�е�w,b,eֵ

���룺
    A,B,Alpha,X,y����ͬLRSVM��ʹ�õ�
�����
    w,b,eΪSVM�еı�����
%}

function [w,b,e] = LRSVM_1_SVM_para(A,B,Alpha,X,y,C)
    N = size(X, 2);

    w = B'*A'*X*(Alpha.*y);
    
    b = 0;
    num = 0;
    for i = 1:N
        if (round(Alpha(i), 6) > 0 && round(Alpha(i), 6) < C)
            b = b + y(i) - w'*B'*A'*X(:,i);
            num = num + 1;
        end
    end
    if (num ~= 0)
        b = b / num;
    end
%     if (num == 0)    %debug
%         disp('û��֧������')
%     end
    
    e = zeros(N,1);   %�ɳڱ���
    for i = 1:N
        if (Alpha(i)-C < 1e-6)
            e(i) = 0;
        else
            e(i) = 1 - y(i)*(w'*B'*A'*X(:,i)+b);
        end
    end
end