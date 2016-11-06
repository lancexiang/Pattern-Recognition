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
            distance(j)=norm(test(i,:)-train(j,:));%ȡŷ�Ͼ���,�õ�����������ѵ��������ŷ�Ͼ���
        end
        [~,train_position]=sort(distance);%����ŷ�Ͼ����С�����������
        train_position=train_position(1,1:k);%ȡǰk��������ԭ���ݵ�λ��
        train_sign=train_number(train_position,1);%ȡ����ǩ
        table=tabulate(train_sign);
        [number,Index]=max(table(:,2));%�õ�Ƶ����ߵ����
        sign=table(Index,1);
        test_sign=test_number(i,1);
        if(sign==test_sign)
            true=true+1;
        end
    end
    truerate1(1,ii)=true/669;
    totalsum=totalsum+truerate1(1,ii);
    fprintf('��%d��%d����ʶ��usps����ȷ��Ϊ%4.2f\n',ii,k,truerate1(1,ii));
    k=k+2;
    x=x+669;
    y=y+669;
end
fprintf('%d��K����ʶ��usps��ƽ����ȷ��Ϊ%4.2f\n',ii,totalsum/ii);





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
truerate2=zeros(1,3);
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
    truerate2(1,ii)=true/10;
    totalsum=totalsum+truerate2(1,ii);
    fprintf('��%d��%d����ʶ��iris����ȷ��Ϊ%4.2f\n',ii,k,truerate2(1,ii));
    k=k+2;
end
fprintf('%d��K����ʶ��iris��ƽ����ȷ��Ϊ%4.2f\n',ii,totalsum/ii);




load sonar
data_w1=sonar1; % ���ȡw1�������е����֮�ģ���78��
rbow1=randperm(98);
trainsample_w1=data_w1(rbow1(:,1:78),1:61);
testsample_w1=data_w1(rbow1(:,79:98),1:61); %ʣ���20����Ϊ��������
data_w2=sonar2; %���ȡw2�������е����֮�ģ���88��
rbow2=randperm(110);
trainsample_w2=data_w2(rbow2(:,1:88),1:61);
testsample_w2=data_w2(rbow2(:,89:110),1:61);%ʣ���22����Ϊ��������
trainsample=cat(1,trainsample_w1,trainsample_w2); %������ѵ�������ϲ�Ϊһ����Ϊ��������֮�ģ���166������
testsample=cat(1,testsample_w1,testsample_w2);%���������������ϲ�Ϊһ����Ϊ������������֮һ����42������
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
        [~,train_position]=sort(distance);%����ŷ�Ͼ����С�����������
        train_position=train_position(1,1:k);%ȡǰk��������ԭ���ݵ�λ��
        train_sign=trainsample(train_position,61);%ȡ����ǩ
        table=tabulate(train_sign);
        [number,Index]=max(table(:,2));%�õ�Ƶ����ߵ����
        sign=table(Index,1);
        test_sign=testsample(i,61);
        if(test_sign==sign)
            true=true+1;
        end
    end
    x=x+14;y=y+14;
    truerate3(1,ii)=true/14;
    totalsum=totalsum+truerate3(1,ii);
    fprintf('��%d��%d����ʶ��sonar����ȷ��Ϊ%4.2f\n',ii,k,truerate3(1,ii));
    k=k+2;
end
fprintf('%d��K����ʶ��sonar��ƽ����ȷ��Ϊ%4.2f\n',ii,totalsum/ii);

figure(1)%��ͼ����
k1=1:3;
plot(truerate1(1,k1),'*')
hold on
plot(truerate2(1,k1),'o')
hold on 
plot(truerate3(1,k1),'x')
hold on 
legend('Fisher��usps�б�','Fisher��iris�б�','Fisher��sonar�б�');
xlabel('ʵ�����');
ylabel('׼ȷ��%');
grid on;
axis([0 3 0 1]);
title('knn���η����׼ȷ��');
