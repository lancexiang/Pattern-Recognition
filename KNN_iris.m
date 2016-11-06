clc;clear;
load iris
rbow1=randperm(50);
trainsample_w1=iris1(rbow1(:,1:40),1:5);% ���ȡw1�������е����֮�ģ���40��
testsample_w1=iris1(rbow1(:,41:50),1:5);%ʣ���10����Ϊ��������
rbow1=randperm(50);
trainsample_w2=iris2(rbow1(:,1:40),1:5);%���ȡw2�������е����֮�ģ���40��
testsample_w2=iris2(rbow1(:,41:50),1:5);%ʣ���10����Ϊ��������
rbow1=randperm(50);
trainsample_w3=iris3(rbow1(:,1:40),1:5);%���ȡw3�������е����֮�ģ���40��
testsample_w3=iris3(rbow1(:,41:50),1:5);%ʣ���10����Ϊ��������
trainsample=cat(1,trainsample_w1,trainsample_w2,trainsample_w3); %������ѵ�������ϲ�Ϊһ��������������֮�ģ���120������
testsample=cat(1,testsample_w1,testsample_w2,testsample_w3);%���������������ϲ�Ϊһ��������������֮һ����30������
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
        [~,train_position]=sort(distance);%����ŷ�Ͼ����С�����������
        train_position=train_position(1,1:k);%ȡǰk��������ԭ���ݵ�λ��
        train_sign=trainsample(train_position,5);%ȡ����ǩ
        table=tabulate(train_sign);
        [number,Index]=max(table(:,2));%�õ�Ƶ����ߵ����
        sign=table(Index,1);
        test_sign=testsample(i,5);
        if(test_sign==sign)
            true=true+1;
        end
    end
    x=x+10;y=y+10;
    truerate(1,ii)=true/10;
    totalsum=totalsum+truerate(1,ii);
    fprintf('��%d��%d����ʶ��iris����ȷ��Ϊ%4.2f\n',ii,k,truerate(1,ii));
    k=k+2;
end
fprintf('%d��K����ʶ��iris��ƽ����ȷ��Ϊ%4.2f\n',ii,totalsum/ii);
figure(1)%��ͼ����
k2=1:3;
plot(k2,truerate(1,k2),'o');
hold on 
legend('KNN����ڶ�irisʶ��');
xlabel('ʵ�����');
ylabel('׼ȷ��%');
grid on;
axis([0 3 0 1]);
title('iris���η����׼ȷ��');