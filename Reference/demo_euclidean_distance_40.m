clear all
close all

% Step 1: Image database input
% cd '******\your directory'
cd 'C:\Users\user\Desktop\ELEC4543\Student_ ELEC4543 Assignment\Student_ ELEC4543 Assignment'

% Database parameters
% !!! Change the ImagePath to the directory of att_faces_10_people folder !!!
% ImgPath = '*****\att_faces_10_people';
ImgPath = 'C:\Users\user\Desktop\ELEC4543\Student_ ELEC4543 Assignment\Student_ ELEC4543 Assignment\att_faces_40_people'

num_samples = 40;  % Number of Samples/People
num_images_each_sample = 10;  % Number of Photos of each sample
num_train = 5;  % Number of Trained Date
num_test = 5;  % Number of Test Data
num_train_images = num_samples*num_train;
num_test_images = num_samples*num_test;

template=[];
imSize = [112 92];
for i=1:num_images_each_sample
    for j=1:num_samples
        template(:,:,i,j) = double(imresize(imread(strcat(ImgPath,'\s',num2str(j),'\',num2str(i),'.jpg')),imSize))/255;% use the set from 1 to 10
    end
end

% build train image matrix
train_images = [];  train_outputs = []; train_labels = [];
for i=1:num_train
    for j=1:num_samples
       [row col] = size(template(:,:,i,j));  
       train_images=[train_images reshape((template(:,:,i,j)), row*col,1)]; %reshape the matrix to a column vector
       train_outputs(j,j+num_samples*(i-1)) = 1;
       train_labels = [train_labels j];
    end
end
% 
% figure
% imshow(reshape( train_images(:,2),imSize));


% build test image matrix
test_images = [];  test_labels = [];
for i = (1 + num_train):(num_train + num_test)
    for j=1:num_samples
       [row col] = size(template(:,:,i,j));  
       test_images=[test_images reshape((template(:,:,i,j)), row*col,1)]; %reshape the matrix to a column vector
       test_labels = [test_labels j];
    end
end

% Step 2: Use the Eigenface method to reduce the input dimensions
% Step 2.1: Calcuate the mean image of the train inputs 
mean_image = mean(train_images')';
% Step 2.2: Determine the diffences matrix: A
% A = train_images - mean_image;
A = train_images - repmat(mean_image,1,num_train_images);
% Step 2.3: Obtain the eigenvectors and eigenvalues of A
%           A'*A is the covariance matrix of A
[V, D] = eig(A'*A);
% Note that all the eigenvectors are already normalized to unit vectors
% The first eigenvalue is zero and is set to 1 for numerical reason
D(1,1) = 1;

% Step 2.4: Choose the best 95% of eigenvalues as the new reduced dimension
eigval=diag(D);
eigsum = sum(eigval); 
csum = 0; 
for i = num_train_images:-1:1 
    csum = csum + eigval(i); 
    tv = csum/eigsum; 
    if tv > 0.95
        k95 = i;
    break 
    end 
end
sprintf(' The number of eigenvalues is %d',num_train_images)
sprintf('Keep the index from %d to %d',k95,num_train_images)
sprintf('The last %d are kept',(num_train_images-k95+1))


% Step 3: Determine the weights with reference to the set of eigenafces U 
% Use the last k95 componments
i_start= k95;
i_end = num_train_images;
% Obtain the ranked eigenfaces Ur
Ur = A*V(:,i_start:i_end);  
% Obtain the ranked eigen values Dr
Dr = D(i_start:i_end,i_start:i_end);
% Obtaint the eigen weight martix:EigenWeights
EigenWeights = inv(Dr)*Ur'*A;

% Step 4; Use the Euclidean Distance measure method
test_result = [];

% !!! --------------- Your Code Start ------------------------ !!!

% Constant & Initialization
Kohonen_node_num = 40;
input_units_num = num_train_images-k95+1;
maxiteration = 100;
learnrate = 0.3;

weights = [];
% weights = rand(Kohonen_node_num, input_units_num);
% for i = 1:Kohonen_node_num
%    weights = [weights, EigenWeights(:,i)];
%end
%weights = weights';

% Center of Clusters by mean of train samples
for i = 1:Kohonen_node_num
    tmp_eigenWeights = zeros(input_units_num, 1);
    for j = i:10:num_train_images
        tmp_eigenWeights = tmp_eigenWeights + EigenWeights(:,j);
    end
    weights(i,:) = tmp_eigenWeights/5;
end

weights_mean = weights;

% Start the competitive learning


tic
for iteration = 1:maxiteration
    iteration;
    
    for i = 1:num_train_images
        % Obtain an input data vector
        I = EigenWeights(:,i)';
        
        % Calculate all the outputs, number of input_units_num
        % taking the Euclidean distance between I and the weight vector
        for n = 1:Kohonen_node_num
            tmp_outputs(n) = sqrt(sum((I-weights(n,:)).^2));
        end
        
        % return the minimum distantce, therefore b is the winning node
        [a,b] = min(tmp_outputs);
        
        %sprintf('iteration = %d, i = %d, b = %d, a = %d', iteration, i, b, a)
        
        % Learning for the weights
        weights(b,:) = weights(b,:) + learnrate*(I-weights(b,:));
        
    end
    
end

time = toc


% Test the network
winner = zeros(1, Kohonen_node_num);

test_outputs = [];
for i = 1:num_test_images
        test_image = test_images(:,i);
        eigen_weights_test = inv(Dr)*Ur'*(test_images - repmat(mean_image,1,num_test_images));

        
        % Obtain an input data vector
        I = eigen_weights_test(:,i)';
        
        % Calculate all the outputs, number of input_units_num
        % taking the Euclidean distance between I and the weight vector
        for n = 1:Kohonen_node_num
            outputs(n) = sqrt(sum((I-weights(n,:)).^2));
        end
        
        % return the minimum distantce, therefore B is the winning node
        [a,b] = min(outputs);
        
        winner(b) = winner(b) + 1;
        test_result(i) = b;
        test_outputs = [test_outputs; outputs];
        
end

winner, weights;




% !!! --------------- Your Code Finish ------------------------ !!!

total_errors = sum(test_result ~= test_labels);

sprintf('The number of testing images is %d', num_test_images)
sprintf('The total error of Euclidean method is %d', total_errors)












