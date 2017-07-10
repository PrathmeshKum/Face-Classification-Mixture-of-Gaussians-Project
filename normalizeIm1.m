function outIm = normalizeIm1(IMAGE_MATRIX,IMAGE_NUM)

IMAGE_MATRIX=double(IMAGE_MATRIX);

 for iFile = 1:(IMAGE_NUM-1);
     
     minval = min(IMAGE_MATRIX(:,iFile));
     maxval = max(IMAGE_MATRIX(:,iFile));
     outIm(:,iFile) = ((IMAGE_MATRIX(:,iFile))-minval)/(maxval-minval);
 
 end