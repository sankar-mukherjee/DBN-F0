size1 = [150 150 150; 150 200 250; 200 250 200; 120 120 120; 100 80 100; 250 200 250];

for i=1:size(size1,1)
    dbn_s = size1(i,:)
    dbn_dnn_class_serial(dbn_s)
end