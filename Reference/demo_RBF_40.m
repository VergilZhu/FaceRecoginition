clear all
close all


% Step 1: Image database input
% cd '******\your directory'
% cd 'D:\gpang\rebirth\Teaching\ELEC4543_new\Course Lecture Materials\Class-05-RBF-HOP-SVM\Class05d ANN_Assignment\Student_ ELEC4543 Assignment'
cd 'C:\Users\user\Desktop\ELEC4543\Student_ ELEC4543 Assignment\Student_ ELEC4543 Assignment'

% Database parameters
% !!! Change the ImagePath to the directory of att_faces_10_peoples folder !!!
% ImgPath = 'D:\...\att_faces_10_people';
% ImgPath = 'D:\gpang\rebirth\Teaching\ELEC4543_new\Course Lecture Materials\Class-05-RBF-HOP-SVM\Class05d ANN_Assignment\Student_ ELEC4543 Assignment\att_faces_10_people';
ImgPath = 'C:\Users\user\Desktop\ELEC4543\Student_ ELEC4543 Assignment\Student_ ELEC4543 Assignment\att_faces_40_people'

num_samples = 40;  % Number of Samples/People
num_images_each_sample = 10;  % Number of Photos of each sample
num_train = 5;  % Number of Trained Data
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
% imshow(reshape( train_images(:,1),imSize));


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
        k95 = i
    break 
    end 
end
sprintf(' The number of eigenvalues is %d',num_train_images)
sprintf('Keep the index from %d to %d',k95,num_train_images)
sprintf('The last %d are kept',(num_train_images-k95+1))


% Step 3: Determine the weights with reference to the set of eigenafaces U 
% Use the last k95 componments
i_start= k95;
i_end = num_train_images;
% Obtain the ranked eigenfaces Ur
Ur = A*V(:,i_start:i_end);  
% Obtain the ranked eigen values Dr
Dr = D(i_start:i_end,i_start:i_end);
% Obtaint the eigen weight martix:EigenWeights
EigenWeights = inv(Dr)*Ur'*A;


% Step 4: manually find the centers of each sample (clusters)
% Step: 4.1: Define the center of the clusters
center_of_clusters = [];
for i = 1:num_samples
    mean_sample = mean(EigenWeights(:,[i:10:num_train_images])')';
    center_of_clusters = [center_of_clusters mean_sample];
end


% Step 4.2: Obtain the center of the Gaussian functions
winner = zeros(1,40);
for i = 1:num_train_images
    tmp_outputs = [];
    for n = 1:40
        tmp_outputs(n) = sqrt(sum( ( center_of_clusters(:,n) - EigenWeights(:,i)).^2 ));
    end
    [min_value,min_index] = min(tmp_outputs); % minimum distance is desired
    winner(min_index) = winner(min_index) + 1;
end % end of one iteration

winner


% Step 5. Forming the three RBF units and training using LMS algorithm
tic
maxiteration = 100; 
sigma = 0.3; 
num_clusters = 40;

% Weights of RBF
% Row: index of outputs (10)
% Column: index of cluster (>= 10)
weight_RBF = [];
weight_RBF = rand(num_samples, num_clusters);

Winit = weight_RBF;

for iteration = 1:200
    tmp_error = [];
    for m = 1:num_train_images 
        % Obtain an input data vector
        I = EigenWeights(:,m);
        doutputs = train_outputs(:,m);
        % Calculate the output from the ten hidden units
        for n = 1:num_clusters
            tmp1 = I - center_of_clusters(:,n);
            %tmp2 = tmp1(1)*tmp1(1) + tmp1(2)*tmp1(2);
            houtputs(n,1)=exp( -1*(tmp1'*tmp1)/(2*sigma*sigma));
        end
        % Calculate the output from the three output units
        % weight_RBF: num_sample x num_clusters
        % houtput: num_cluster x 1
        noutputs =  weight_RBF * houtputs;
        error = doutputs - noutputs;
        % weight adjustment using the LMS learning rule
        for i = 1:num_samples
            for j = 1:num_clusters
                weight_RBF(i,j) = weight_RBF(i,j) + 0.5*error(i)*houtputs(j);
            end
        end
        tmp_error = [tmp_error max(abs(error))];
    end % end of one iteration
    % display of the iteration number and error info
    % it should show the convergence and error decreasing
    [iteration mean(tmp_error)];
end % outermost loop

time = toc


% Step.6: Test the RBF network
test_results = [];
for i = 1:num_test_images %test all 10 persons
    test_image = test_images(:,i);
    eigen_weights_test = inv(Dr)*Ur'*(test_image - mean_image);
    for n = 1:num_clusters
        tmp1 = eigen_weights_test - center_of_clusters(:,n);
        houtputs(n,1)=exp( -1*(tmp1'*tmp1)/(2*sigma*sigma));
    end
    result_img = weight_RBF*houtputs;
    test_results = [test_results result_img];
end

total_errors = sum(fcn_convert_to_labels(test_results) ~= test_labels);
sprintf('The number of testing images is %d', num_test_images)
sprintf('The total error of RBF method is %d', total_errors)

% Reconstruct the image of the first person, third image (item 21)
faces = Ur* EigenWeights(:,21)+ mean_image;
figure
imshow(reshape(faces,imSize));

 


