clc;clear;
load usps
k=3;
truerate1=zeros(1,3);
x=1;y=669;
totalsum=0;
distance=zeros(1,7291);
for ii=1:3
    true=0;
    for i=x:y
        for j=1:7291
            distance(j)=norm(test(i,:)-train(j,:));%取欧氏距离,得到测试样本与训练样本的欧氏距离
        end
        [~,train_position]=sort(distance);%排序将欧氏距离从小到大进行排序
        train_position=train_position(1,1:k);%取前k个距离在原数据的位置
        train_sign=train_number(train_position,1);%取出标签
        table=tabulate(train_sign);
        [number,Index]=max(table(:,2));%得到频率最高的类别
        sign=table(Index,1);
        test_sign=test_number(i,1);
        if(sign==test_sign)
            true=true+1;
        end
    end
    truerate1(1,ii)=true/669;
    totalsum=totalsum+truerate1(1,ii);
    fprintf('第%d次%d近邻识别usps的正确率为%4.2f\n',ii,k,truerate1(1,ii));
    k=k+2;
    x=x+669;
    y=y+669;
end
fprintf('%d次K近邻识别usps的平均正确率为%4.2f\n',ii,totalsum/ii);





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
truerate2=zeros(1,3);
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
    truerate2(1,ii)=true/10;
    totalsum=totalsum+truerate2(1,ii);
    fprintf('第%d次%d近邻识别iris的正确率为%4.2f\n',ii,k,truerate2(1,ii));
    k=k+2;
end
fprintf('%d次K近邻识别iris的平均正确率为%4.2f\n',ii,totalsum/ii);




load sonar
data_w1=sonar1; % 随机取w1类数据中的五分之四，即78组
rbow1=randperm(98);
trainsample_w1=data_w1(rbow1(:,1:78),1:61);
testsample_w1=data_w1(rbow1(:,79:98),1:61); %剩余的20组作为测试样本
data_w2=sonar2; %随机取w2类数据中的五分之四，即88组
rbow2=randperm(110);
trainsample_w2=data_w2(rbow2(:,1:88),1:61);
testsample_w2=data_w2(rbow2(:,89:110),1:61);%剩余的22组作为测试样本
trainsample=cat(1,trainsample_w1,trainsample_w2); %将两个训练样本合并为一个作为整体的五分之四，即166组数据
testsample=cat(1,testsample_w1,testsample_w2);%将两个测试样本合并为一个作为包含整体的五分之一，即42组数据
k=3;
totalsum=0;
truerate3=zeros(1,3);
distance=zeros(1,120);
x=1;y=14;
for ii=1:3
    true=0;
    for i=x:y
        for j=1:166
            distance(j)=norm(testsample(i,1:60)-trainsample(j,1:60));
        end
        [~,train_position]=sort(distance);%排序将欧氏距离从小到大进行排序
        train_position=train_position(1,1:k);%取前k个距离在原数据的位置
        train_sign=trainsample(train_position,61);%取出标签
        table=tabulate(train_sign);
        [number,Index]=max(table(:,2));%得到频率最高的类别
        sign=table(Index,1);
        test_sign=testsample(i,61);
        if(test_sign==sign)
            true=true+1;
        end
    end
    x=x+14;y=y+14;
    truerate3(1,ii)=true/14;
    totalsum=totalsum+truerate3(1,ii);
    fprintf('第%d次%d近邻识别sonar的正确率为%4.2f\n',ii,k,truerate3(1,ii));
    k=k+2;
end
fprintf('%d次K近邻识别sonar的平均正确率为%4.2f\n',ii,totalsum/ii);

figure(1)%画图程序
k1=1:3;
plot(truerate1(1,k1),'*')
hold on
plot(truerate2(1,k1),'o')
hold on 
plot(truerate3(1,k1),'x')
hold on 
legend('Fisher对usps判别','Fisher对iris判别','Fisher对sonar判别');
xlabel('实验次数');
ylabel('准确率%');
grid on;
axis([0 3 0 1]);
title('knn三次分类的准确率');
