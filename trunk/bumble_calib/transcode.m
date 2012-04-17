function transcode

for i=1:32
    
    if i<10
        filename1=['cal_left0' int2str(i) '.ppm'];
        filename2=['cal_right0' int2str(i) '.ppm'];
    else
        filename1=['cal_left' int2str(i) '.ppm'];
        filename2=['cal_right' int2str(i) '.ppm'];
    end
    
    if exist(filename1,'file')
        image1=imread(filename1);
        image1=rgb2gray(image1);
        filename1=['cal_left' int2str(i) '.jpg'];
        imwrite(image1,filename1,'JPEG');
    end
    
    if exist(filename2,'file')
        image2=imread(filename2);
        image2=rgb2gray(image2);
        filename2=['cal_right' int2str(i) '.jpg'];
        imwrite(image2,filename2,'JPEG');
    end
       
end