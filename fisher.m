clc;clear;
load iris %��ȡ���ݣ��зֺ��˵�3�����ݣ�ÿ��Ϊ50�飬ÿ�����ݶ�������ǩ
totalsum=0;
truerate=zeros(1,10);%���ڱ���ÿ�ε�������ȷ��
for ii=1:10
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
    true=0;error=0;
    m1=zeros(1,4);%w1��w2��w3�ľ�ֵ����
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
    s1=zeros(4,4);%w1��w2��w3��������ɢ�Ⱦ���
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
    Sw12=s1+s2;%��������������ɢ�Ⱦ���
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
            y1=(testsample(zz,1:4))*w12;%��һ���ж�
            if y1>y12
                yy1=(testsample(zz,1:4))*w13;%�ڶ��η���
                if(yy1>y13)
                    true=true+1;
                end
            end
        elseif(flg==2)
            y1=(testsample(zz,1:4))*w12;%��һ���ж�
            if y1<y12
                yy1=(testsample(zz,1:4))*w23;%�ڶ��η���
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
    fprintf('��%d��Fisher��iris����ʶ����Ϊ%4.2f\n',ii,truerate(1,ii));
    totalsum=totalsum+truerate(1,ii);
end
fprintf('10��Fisher��iris����ƽ��ʶ����Ϊ%4.2f\n',totalsum/10);
figure(1)%��ͼ����
yy1=1:10;
plot(yy1,truerate(1,yy1),'k-');
hold on;
xlabel('ʵ�����');
ylabel('׼ȷ��%');
grid on;
axis([0 10 0 1]);
title('fisherʮ�η����׼ȷ��');


load sonar%��ȡ���ݣ��зֺ��˵�2�����ݣ�һ��Ϊ98�飬��һ��Ϊ110�飬ÿ�����ݶ�������ǩ
totalsum=0;
truerate=zeros(1,10);%��������10�ε�����������ȷ��
for ii=1:10  %����10�ε���
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
    m1=zeros(1,60); %w1��w2�ľ�ֵ����
    m2=zeros(1,60);
    true=0;error=0;
    for y1=1:78
        for yy1=1:60
            m1(1,yy1)=m1(1,yy1)+trainsample(y1,yy1);
        end;
    end;
    m1=m1/78;                                     %��w1��ѵ������ƽ��ֵ
    for y2=79:166
        for yy2=1:60
            m2(1,yy2)=m2(1,yy2)+trainsample(y2,yy2);
        end;
    end;
    m2=m2/88;                                     %��w2��ѵ������ƽ��ֵ
    s1=zeros(60,60);%w1��w2��������ɢ�Ⱦ���
    s2=zeros(60,60);
    for xx=1:78
        s1=s1+((trainsample_w1(xx,1:60)-m1)'*(trainsample_w1(xx,1:60)-m1));
    end;
    for yy=1:88
        s2=s2+((trainsample_w2(yy,1:60)-m2)'*(trainsample_w2(yy,1:60)-m2));
    end;
    Sw=s1+s2;%��������ɢ�Ⱦ���
    w=Sw\(m1-m2)'; %ͶӰ����,����matlab��ʾ���ֱ��ʽ���죬�ʸı��˹�ʽ�����ӣ���inv(Sw)*(m1-m2)'Ч����ͬ
    y0=(m1*w+m2*w)/2;%�ֽ���ֵ��
    for zz=1:42
        y=(testsample(zz,1:60))*w;%����������ͶӰ���бȽ�
        if y>y0
            flg=1;
        else
            flg=2;
        end;
        if(flg==testsample(zz,61))
            true=true+1;
        else
            error=error+1;
        end;
    end;
    truerate(1,ii)=true/42;
    fprintf('��%d��Fisher��sonar����ʶ����Ϊ%4.2f\n',ii,truerate(1,ii));
    totalsum=totalsum+truerate(1,ii);
end;     
fprintf('10��Fisher��sonar����ƽ��ʶ����Ϊ%4.2f\n',totalsum/10);
figure(1)%��ͼ����
k2=1:10;
plot(k2,truerate(1,k2),'r-');
hold on;
legend('Fisher��iris�����б�','Fisher��sonar�����б�');
xlabel('ʵ�����');
ylabel('׼ȷ��%');
grid on;
axis([0 10 0 1]);
title('fisherʮ�η����׼ȷ��');

