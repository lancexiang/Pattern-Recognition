clc;clear;
load iris %读取数据，有分好了的3类数据，每类为50组，每组数据都包含标签
totalsum=0;
truerate=zeros(1,10);%用于保存每次迭代的正确率
for ii=1:10
    rbow1=randperm(50);
    trainsample_w1=iris1(rbow1(:,1:40),1:5);% 随机取w1类数据中的五分之四，即40组
    testsample_w1=iris1(rbow1(:,41:50),1:5);%剩余的10组作为测试样本
    rbow1=randperm(50);
    trainsample_w2=iris2(rbow1(:,1:40),1:5);%随机取w2类数据中的五分之四，即40组
    testsample_w2=iris2(rbow1(:,41:50),1:5);%剩余的10组作为测试样本
    rbow1=randperm(50);
    trainsample_w3=iris3(rbow1(:,1:40),1:5);%随机取w3类数据中的五分之四，即40组
    testsample_w3=iris3(rbow1(:,41:50),1:5);%剩余的10组作为测试样本
    trainsample=cat(1,trainsample_w1,trainsample_w2,trainsample_w3); %将三个训练样本合并为一个包含整体的五分之四，即120组数据
    testsample=cat(1,testsample_w1,testsample_w2,testsample_w3);%将三个测试样本合并为一个包含整体的五分之一，即30组数据
    true=0;error=0;
    m1=zeros(1,4);%w1和w2和w3的均值向量
    m2=zeros(1,4);
    m3=zeros(1,4);
    for y1=1:40
        for yy1=1:4
            m1(1,yy1)=m1(1,yy1)+trainsample(y1,yy1);
        end
    end
    m1=m1/40;
    for y2=41:80
        for yy2=1:4
            m2(1,yy2)=m2(1,yy2)+trainsample(y2,yy2);
        end
    end
    m2=m2/40;
    for y3=81:120
        for yy3=1:4
            m3(1,yy3)=m3(1,yy3)+trainsample(y3,yy3);
        end
    end
    m3=m3/40;
    s1=zeros(4,4);%w1和w2和w3的类内离散度矩阵
    s2=zeros(4,4);
    s3=zeros(4,4);
    for xx=1:40
        s1=s1+((trainsample_w1(xx,1:4)-m1)'*(trainsample_w1(xx,1:4)-m1));
    end;
    for xx=1:40
        s2=s2+((trainsample_w2(xx,1:4)-m2)'*(trainsample_w2(xx,1:4)-m2));
    end;
    for xx=1:40
        s3=s3+((trainsample_w3(xx,1:4)-m3)'*(trainsample_w3(xx,1:4)-m3));
    end;
    Sw12=s1+s2;%两两的总类内离散度矩阵
    Sw13=s1+s3;
    Sw23=s2+s3;
    w12=(inv(Sw12))*(m1-m2)';
    w13=(inv(Sw13))*(m1-m3)';
    w23=(inv(Sw23))*(m2-m3)';
    y12=(m1*w12+m2*w12)/2;
    y13=(m1*w13+m3*w13)/2;
    y23=(m2*w23+m3*w23)/2;
    for zz=1:30
        flg=testsample(zz,5);
        if(flg==1)
            y1=(testsample(zz,1:4))*w12;%第一次判断
            if y1>y12
                yy1=(testsample(zz,1:4))*w13;%第二次分类
                if(yy1>y13)
                    true=true+1;
                end
            end
        elseif(flg==2)
            y1=(testsample(zz,1:4))*w12;%第一次判断
            if y1<y12
                yy1=(testsample(zz,1:4))*w23;%第二次分类
                if(yy1>y23)
                    true=true+1;
                end
            end
        else
            y1=(testsample(zz,1:4))*w13;
            if y1<y13
                yy1=(testsample(zz,1:4))*w23;
                if(yy1<y23)
                    true=true+1;
                end
            end
        end
    end
    truerate(1,ii)=true/30;
    fprintf('第%d次Fisher分类识别率为%4.2f\n',ii,truerate(1,ii));
    totalsum=totalsum+truerate(1,ii);
end
fprintf('10次Fisher分类平均识别率为%4.2f\n',totalsum/10);
figure(1)
linetype={'r-'};%画图程序
yy1=1:10;
plot(yy1,truerate(1,yy1),linetype{1});
hold on;
legend('Fisher对iris线性判别');
xlabel('实验次数');
ylabel('准确率%');
grid on;
axis([0 10 0 1]);
title('iris十次分类的准确率');
