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
   