function [lambda, mean, sigma] = MOG_FCN (image_matrix, K, precision, image_num, SIZE)
         

 % Initialize all values in lambda to 1/K.
 
    lambda = repmat (1/K, K, 1);  % K times row & same number of columns
    
 % Initialize the values in mean matrix to K randomly chosen unique image vectors.
 
    I = size (image_matrix, 1);
    K_random_unique_integers = randperm(I);
    K_random_unique_integers = K_random_unique_integers(1:K);
    mean = image_matrix (K_random_unique_integers,:);
    mean=double(mean);
    
 % Initialize the variances in sigma to the variance of the image matrix.
 
    sigma = cell (1, K);
    dimensionality = size (image_matrix,2);
    dataset_mean = sum(image_matrix,1) ./ I;
    dataset_variance = zeros (dimensionality, dimensionality);
    
    for i = 1 : I
        mat = double(image_matrix (i,:)) - dataset_mean;
        mat = mat' * mat;
        dataset_variance = dataset_variance + mat;
    end
    dataset_variance = dataset_variance ./ I;
    
    for i = 1 : K
        sigma{i} = dataset_variance;
    end
    
  % The main loop.
    iterations = 0;
    disp(iterations);
    previous_L = 1000000; % just a random initialization
    while true
        % Expectation step.
        l = zeros (I,K);
        r = zeros (I,K);
        % Compute the numerator of Bayes' rule.
        for k = 1 : K
            
            sigma{k}=diag(sigma{k});
            sigma{k}=diag(sigma{k}); % For creating matrix with elements only along the diagonal 
            l(:,k) =lambda(k) * mvgd1(image_matrix, mean(k,:), sigma{k}, image_num, SIZE);
        end
        
        % Compute the responsibilities by normalizing.
        s = sum(l,2);        
        for i = 1 : I
            r(i,:) = l(i,:) ./ s(i);
        end

        % Maximization step.
        r_summed_rows = sum (r,1);
        r_summed_all = sum(sum(r,1),2);
        for k = 1 : K
            % Update lambda.
            lambda(k) = r_summed_rows(k) / r_summed_all;

            % Update mean.
            new_mean = zeros (1,dimensionality);
            for i = 1 : I
                new_mean = new_mean + r(i,k)*image_matrix(i,:);
            end
            mean(k,:) = new_mean ./ r_summed_rows(k);

            % Update sigma.
            new_sigma = zeros (dimensionality,dimensionality);
            for i = 1 : I
                mat = image_matrix(i,:) - mean(k,:);
                mat = r(i,k) * (mat' * mat);
                new_sigma = new_sigma + mat;
            end
            sigma{k} = new_sigma ./ r_summed_rows(k);
        end
        
        % Compute the log likelihood L.
        temp = zeros (I,K);
        for k = 1 : K
            temp(:,k) = lambda(k)*mvgd1(image_matrix, mean(k,:), sigma{k}, image_num, SIZE);
        end
        temp = sum(temp,2);
        temp = log(temp);        
        L = sum(temp);
        L=abs(L);
        disp(L);
 
        iterations = iterations + 1;        
        disp([num2str(iterations) ': ' num2str(L)]);
        
        if abs(L - previous_L) < precision
            %msg = [num2str(iterations) ' iterations, log-likelihood = ', ...
                %num2str(L)];
            %disp(msg);
            disp(' EM algorithm executed! ');
            break;
        end
        
        previous_L = L;
        
    end
end
    