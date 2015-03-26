clear;clc;
load('data/f0.mat');
%f0_5state = reshape(f0_5state',1,[])';
s=size(f0_5state,1);
z=f0_5state;
delta=zeros(s,1);
delta_delta=zeros(s,1);
for i=1:s
   c=z(i);
   if(i==1)
       p=0;
   else
       p=z(i-1);
   end
   if(i==s)
       n=0;
   else
       n=z(i+1);
   end
   delta(i)=n-p;
   delta_delta(i)=0.5*n-2*c+0.5*p;
end