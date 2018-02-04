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

tic
% Step 4: Train the SVM Classifier
SVMModels = cell(num_samples,1);  % Define the 10 SVM models
classes = unique(train_labels);    % Get the value of each class

for j = 1:numel(classes)
    % If the Y is to the classes(j), the value of index is set to 1;
    % otherwise, the value is set to 0;
    indx = (train_labels == classes(j));   
    % Create binary classes for each classifier
    % 'KernelScale' is set to 'auto'
    SVMModels{j} = fitcsvm(EigenWeights',indx','KernelFunction','rbf','Standardize',true,...
       'KernelScale','auto','ClassNames',[false true]);
%     SVMModels{j} = fitcsvm(EigenWeights',indx','ClassNames',[false true],'Standardize',true,...
%         'KernelFunction','rbf','BoxConstraint',1);
end
time = toc
% Step 5: Predict the classes of the test images
test_results = [];
eigen_weights_test = inv(Dr)*Ur'*(test_images - repmat(mean_image,1,num_train_images));


%eigen_weights_test = inv(Dr)*Ur'*(train_images - mean_image);
% Define the socres matrix to store the class values of each input
Scores = zeros(size(eigen_weights_test,2),numel(classes));
for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},eigen_weights_test');
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end
test_results = [test_results Scores];
Scores;
[~,maxScore] = max(Scores,[],2);

total_errors = sum(maxScore' ~= test_labels);
sprintf('The number of testing images is %d', num_test_images)
sprintf('The total error of SVM method is %d', total_errors)



