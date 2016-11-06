clear;
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
