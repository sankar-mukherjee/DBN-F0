clear;
load('data/f0.mat');load('data/input.mat');

train_yy = zeros(size(f0_5state,1),327); %max 400 so 400-73=327
for i=1:size(f0_5state)
    c=zeros(1,327);
    t=ceil(f0_5state(i,:));
    if(t==0)
        c(1)=1;
    else
        c(t-73)=1;   %min is 75 so 75-73=2 as 1 is researved for 0
    end
    train_yy(i,:)=c;
end
f0_5state = train_yy;

train_xx = expand(input,[5,1]);
z=[1; 2 ;3 ;4 ;5];
x=repmat(z,[2301230/5 , 1]);
z=zeros(size(train_xx,1),5);
for i=1:size(train_xx,1)
   z(i,x(i))=1; 
end
x=[train_xx z];
input = x;
% save('data/input_class.mat','input');
%save('data/f0_class.mat','f0_5state');