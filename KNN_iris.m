clc;clear;
load iris
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
k=3;
totalsum=0;
truerate=zeros(1,3);
distance=zeros(1,120);
x=1;y=10;
for ii=1:3
    true=0;
    for i=x:y
        for j=1:120
            distance(j)=norm(testsample(i,1:4)-trainsample(j,1:4));
        end
        [~,train_position]=sort(distance);%排序将欧氏距离从小到大进行排序
        train_position=train_position(1,1:k);%取前k个距离在原数据的位置
        train_sign=trainsample(train_position,5);%取出标签
        table=tabulate(train_sign);
        [number,Index]=max(table(:,2));%得到频率最高的类别
        sign=table(Index,1);
        test_sign=testsample(i,5);
        if(test_sign==sign)
            true=true+1;
        end
    end
    x=x+10;y=y+10;
    truerate(1,ii)=true/10;
    totalsum=totalsum+truerate(1,ii);
    fprintf('第%d次%d近邻识别iris的正确率为%4.2f\n',ii,k,truerate(1,ii));
    k=k+2;
end
fprintf('%d次K近邻识别iris的平均正确率为%4.2f\n',ii,totalsum/ii);
figure(1)%画图程序
k2=1:3;
plot(k2,truerate(1,k2),'o');
hold on 
legend('KNN最近邻对iris识别');
xlabel('实验次数');
ylabel('准确率%');
grid on;
axis([0 3 0 1]);
title('iris三次分类的准确率');