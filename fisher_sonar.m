clc;clear;
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
legend('Fisher��sonar�����б�');
xlabel('ʵ�����');
ylabel('׼ȷ��%');
grid on;
axis([0 10 0 1]);
title('sonarʮ�η����׼ȷ��');