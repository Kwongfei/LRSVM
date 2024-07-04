%{
README.txt:
LRSVM_1_2_Recognition(train, test, train_labels, test_labels, Lambda, C, s)
设计迭代算法实现LRSVMOA,并且对测试集作识别。

输入：
    train, test：训练集和测试集，每一列是一个数据
    train_labels, test_labels：标签，字典形式。
    Lambda: 回归项系数
    C：松弛变量的惩罚项
    s：秩
输出：
train_error_rate: 训练集错误率
test_error_rate: 测试集错误率
Fval： 目标函数值

cite:
Liang G, Lai Z, Kong H. Support vector machine with discriminative low‐rank embedding. CAAI Transactions on Intelligence Technology, 2024.
%}

function [train_error_rate, test_error_rate, Fval] = LRSVM_1_2_Recognition(train, test, train_labels, test_labels, Lambda, C, s)
    d = size(train,1);
    N = size(train, 2);

    avg = mean(train,2);
    train = train - avg*ones(1,N);
    test = test - avg*ones(1,size(test,2));
    y_train = train_labels*[1;-1];
    y_test = test_labels*[1;-1];
    
    X = train;
    y = y_train;
    Y = train_labels;
    c = size(Y,2);
    rng('default');
    rng(1);
    A = rand(d,s);
    B = rand(s, c);
    H1 = (B'*A'*X).*(ones(c,1)*y');
    H = H1'*H1/2;
    H = (H+H')/2;
    f = -ones(N,1);
    lb = zeros(N,1);
    ub = C*ones(size(lb));
    Alpha = quadprog(H,f,[],[],[],[],lb,ub);    %根据随机生成的A生成Alpha
    [w,~,e] = LRSVM_1_SVM_para(A,B,Alpha,X,y,C);
    Fval0 = w'*w/2+Lambda*norm(Y-X'*A*B,'fro')+C*sum(e);
    
    T_max = 30;
    T = 0;
    Fval = zeros(T_max,1);
    
    Alpha_all = zeros(N,T_max);
    while (T < T_max)
        T = T + 1;
        disp('当前迭代次数：')
        disp(T)
        % 迭代更新矩阵A,B
        H = (X*(eye(N)-(Alpha.*y)*(Alpha.*y)'/(2*Lambda))*X'+0.00001*eye(d))\((X*Y)*(Y'*X'));
        [EV, D] = eig((H+H') / 2);
        [~, ind] = sort(diag(D),'descend');
        EVs = EV(:,ind);
        A = EVs(:,1:s);
        B = (A'*X*(2*eye(N)-(Alpha.*y)*(Alpha.*y)'/Lambda)*X'*A+0.00001*eye(s))\(2*A'*X*Y);
        
%         disp('求解Alpha')
        % 迭代更新alpha
        H1 = (B'*A'*X).*(ones(c,1)*y');
        H = H1'*H1/2;
        H = (H+H')/2;
        f = -ones(N,1);
        lb = zeros(N,1);
        ub = C*ones(size(lb));
        Alpha = quadprog(H,f,[],[],[],[],lb,ub);    %根据随机生成的A生成Alpha
        
        Alpha_all(:,T) = Alpha;
        
        [w,~,e] = LRSVM_1_SVM_para(A,B,Alpha,X,y,C);
        
        Fval(T) = w'*w/2+Lambda*norm(Y-X'*A*B,'fro')+C*sum(e);
        if (T > 2 && (abs(Fval(T-1)-Fval(T))/abs(Fval(T)) < 1e-6 || abs(Fval(T-2)-Fval(T))/abs(Fval(T)) < 1e-6))
            break
        end
    end
    
%     plot(Fval);
%     title('目标函数值')
    Fval = [Fval0;Fval(1:T)];
    
    [w,b,~] = LRSVM_1_SVM_para(A,B,Alpha,X,y,C);
    
    Good = 0;
    ind_train = zeros(size(train,2),1);
    for i = 1:size(train,2)
        y0 = w'*B'*A'*train(:,i)+b;
        ind_train(i) = y0;
        if (y0*y_train(i) > 0)
            Good = Good + 1;
        end
    end
    train_error_rate = 1 - Good / size(train,2);

    Good = 0;
    ind_test = zeros(size(test,2),1);
    for i = 1:size(test,2)
        y0 = w'*B'*A'*test(:,i)+b;
        ind_test(i) = y0;
        if (y0*y_test(i) > 0)
            Good = Good + 1;
        end
    end
    test_error_rate = 1 - Good / size(test,2);
    
end
