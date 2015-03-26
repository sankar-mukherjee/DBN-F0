function original=add_delta_deltadelta(z,dz,ddz)
% generation of f0 by adding delta and delta-delta to the static feature
% delta c = c(t+1) - c(t-1)
% delta-delta c = 0.5c(t+1) - 2c(t) + 0.5c(t-1)
% so closed form is O = WC where O = [c, delta c, delta-delta c]
w=[0 1 0;-1 0 1;0.5 -2 0.5]; % W
x=zeros(size(z));
for i=1:length(z)
   P = [z(i) dz(i) ddz(i)];
   O = P * inv(w);
   x(i)=O(2);
end
original=x;