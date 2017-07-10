clc;
clear all;
close all;
warning('off');

disp(' ## Program for Face Classification by MOG ## ');
 
%file directory
    
directory=char(pwd);
TrainingfcDirectory = 'face_train_resized\';
TrainingbgDirectory = 'background_train_resized\';
TestingfcDirectory = 'face_test_resized\';
TestingbgDirectory = 'background_test_resized\';

TrainingfcFiles = dir(TrainingfcDirectory);
TrainingbgFiles = dir(TrainingbgDirectory);
TestingfcFiles = dir(TestingfcDirectory);
TestingbgFiles = dir(TestingbgDirectory);

fc_image_num=1;

for iFile = 3:size(TrainingfcFiles,1);
     
    %loading the image and converting into vector
    
    origIm=imread([TrainingfcDirectory TrainingfcFiles(iFile).name]);
    origIm = imresize(origIm,0.5);
    vIm=reshape(origIm,[1 900]);
    face_matrix(fc_image_num,:)=vIm;
    fc_image_num=fc_image_num+1;
    
end

face_matrix=normalizeIm1((face_matrix'),fc_image_num);
face_matrix=face_matrix';
disp('Computing EM Algorithm for Training Face Images');
[lambda_face, mean_face, sigma_face] = MOG_FCN(face_matrix, 3, 100, fc_image_num, 900); % For 3 Gaussian mixture curvature.



bg_image_num=1;

for iFile = 3:size(TrainingbgFiles,1);
     
    %loading the image and converting into vector
    
    origIm=imread([TrainingbgDirectory TrainingbgFiles(iFile).name]);
    origIm = imresize(origIm,0.5);
    vIm=reshape(origIm,[1 900]);
    background_matrix(bg_image_num,:)=vIm;
    bg_image_num=bg_image_num+1;
    
end

background_matrix=normalizeIm1((background_matrix'),bg_image_num);
background_matrix=background_matrix';
disp('Computing EM Algorithm for Training Background Images');
[lambda_bg, mean_bg, sigma_bg] = MOG_FCN(background_matrix, 3, 250, bg_image_num, 900); % For 3 Gaussian mixture curvature.

% INFERENCE ALGORITHM:


True_fc_num=0;
False_fc_num=0;

fc_test_image_num=1;

for iFile = 3:size(TestingfcFiles,1);
     
    %loading the image and converting into vector
    
    origIm=imread([TestingfcDirectory TestingfcFiles(iFile).name]);
    origIm = imresize(origIm,0.5);
    vIm=reshape(origIm,[1 900]);
    test_face_matrix(fc_test_image_num,:)=vIm;
    fc_test_image_num=fc_test_image_num+1;
    
end

test_face_matrix=normalizeIm1((test_face_matrix'),fc_test_image_num);
test_face_matrix=test_face_matrix';

disp('computing inference for test face images: ');

for k = 1 : 3
    
    fc_img_test(:,k)=lambda_face(k)*mvgd1(test_face_matrix, mean_face(k,:), sigma_face{k}, fc_test_image_num, 900);
end
 fc_img_test=double(fc_img_test);
 
 for  iFile = 1:fc_test_image_num-1;
     
     fc_img_test1(iFile,1)=sum(fc_img_test(iFile,:));
     
 end
 
 
 for k = 1 : 3
    
    fc_img_test(:,k)=lambda_bg(k)*mvgd1(test_face_matrix, mean_bg(k,:), sigma_bg{k}, fc_test_image_num, 900);
end
 fc_img_test=double(fc_img_test);
 
 for  iFile = 1:fc_test_image_num-1;
     
     fc_img_test2(iFile,1)=sum(fc_img_test(iFile,:));
     
 end
 
 for iFile = 1:fc_test_image_num-1;
    
    if (fc_img_test1(iFile,1)) > (fc_img_test2(iFile,1));
        
        True_fc_num=True_fc_num+1;
        True_face_images(True_fc_num,1)=iFile; 
        
    else
        
        False_fc_num=False_fc_num+1;
        False_face_images(False_fc_num,1)=iFile;
    
    end
 end

True_bg_num=0;
False_bg_num=0;

bg_test_image_num=1;


for iFile = 3:size(TestingbgFiles,1);
     
    %loading the image and converting into vector
    
    origIm=imread([TestingbgDirectory TestingbgFiles(iFile).name]);
    origIm = imresize(origIm,0.5);
    vIm=reshape(origIm,[1 900]);
    test_background_matrix(bg_test_image_num,:)=vIm;
    bg_test_image_num=bg_test_image_num+1;
    
end

test_background_matrix=normalizeIm1((test_background_matrix'),bg_test_image_num);
test_background_matrix=test_background_matrix';

disp('computing inference for test background images: ');

for k = 1 : 3
    
    bg_img_test(:,k)=lambda_bg(k)*mvgd1(test_background_matrix, mean_bg(k,:), sigma_bg{k}, bg_test_image_num, 900);
end
 bg_img_test=double(bg_img_test);
 
 for  iFile = 1:bg_test_image_num-1;
     
     bg_img_test1(iFile,1)=sum(bg_img_test(iFile,:));
     
 end
 
 
 for k = 1 : 3
    
    bg_img_test(:,k)=lambda_face(k)*mvgd1(test_background_matrix, mean_face(k,:), sigma_face{k}, bg_test_image_num, 900);
end
 bg_img_test=double(bg_img_test);
 
 for  iFile = 1:bg_test_image_num-1;
     
     bg_img_test2(iFile,1)=sum(bg_img_test(iFile,:));
     
 end
 
 
for iFile = 1:bg_test_image_num-1;
    
    if (bg_img_test1(iFile,1)) > (bg_img_test2(iFile,1));
        
        True_bg_num=True_bg_num+1;
        True_bg_images(True_bg_num,1)=iFile;
        
    else
        
        False_bg_num=False_bg_num+1;
        False_bg_images(False_bg_num,1)=iFile;
    
    end
end

% Calculation of accuracy:

disp('computing accuracy: ');

face_accuracy=(True_fc_num*100)/(fc_test_image_num-1);
background_accuracy=(True_bg_num*100)/(bg_test_image_num-1);
total_accuracy=((True_fc_num+True_bg_num)*100)/((fc_test_image_num-1)+(bg_test_image_num-1));
