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
   