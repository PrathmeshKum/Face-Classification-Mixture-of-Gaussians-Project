function dist = mvgd1(IMAGE,MEAN,SIGMA,IMAGE_NUM,SIZE)

mod_sigma =(prod(vpa((sqrt(sum(SIGMA.^2)))'))); 
first_term=((SIZE/2)*log((mod_sigma)));
SIGMA_INVERSE=(inv(SIGMA));


 for iFile = 1:IMAGE_NUM-1;

     summation_term=(((transpose(IMAGE(iFile,:))-transpose(MEAN))*transpose((transpose(IMAGE(iFile,:))-transpose(MEAN)))))*SIGMA_INVERSE;
     dist(iFile,:)=-(first_term)-((0.5)*(sum(summation_term(:))));
    
     %disp(iFile);

 end