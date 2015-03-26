clear;clc;

% s=2000; 
% Fs = 8000;                   % samples per second
%    dt = 1/Fs;                   % seconds per sample
%    StopTime = 0.25;             % seconds
%    t = (0:dt:StopTime-dt)';     % seconds
%    Fc = 60;                     % hertz
% z=(rand(1)*cos(2*pi*Fc*t) + rand(1)*cos(2*pi/3*Fc*t))';    

s=100;
z=rand(1,s);        
dz=zeros(1,s);
ddz=zeros(1,s);
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
   dz(1,i)=n-p;
   ddz(1,i)=0.5*n-2*c+0.5*p;
end

w=[0 1 0;-1 0 1;0.5 -2 0.5];
x=zeros(1,s);
for i=1:s
   P = [z(i) dz(i) ddz(i)];
   O = P * inv(w);
   x(1,i)=O(2);
end
rmse(-z,x)
plot(x,'DisplayName','x','YDataSource','x');hold all;plot(-z,'DisplayName','z','YDataSource','z');hold off;figure(gcf);
