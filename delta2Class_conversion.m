clear;clc;
%% delta to class conversion 80 for delta [10 resolution]
%  and 139 for delta_delta [8 resolution]

load('data/delta_f0.mat');
z=unique(ceil(delta));
z=reshape(z,[10 80])';
D = zeros(size(delta,1),80);
for i=1:size(delta,1)
    c=zeros(1,80);
    [s, ~]=find(z==ceil(delta(i)));    
    c(s)=1;
    D(i,:) = c;
end

z=unique(ceil(delta_delta));
z=reshape(z,[8 139])';
DD = zeros(size(delta,1),139);
for i=1:size(delta_delta,1)
    c=zeros(1,139);
    [s, ~]=find(z==ceil(delta_delta(i)));    
    c(s)=1;
    DD(i,:) = c;
end
delta=D;delta_delta=DD;