%% This code is used to test which vesicles are being detected in individual images %%
%Use this as a quick check to assess whether there is sufficient contrast between your
%vesicles and the background to detect the edges, and assign suitable
%values for the sensitivity_factor and min/max diameter variables. The
%'marker images' used in this work were Texas Red labelled vesicles - these images were taken using the TXR channel on 
%the microscope and had the _TXR.png file extension. 'Detection 
%images' used in this work were the same field of view as the TXR image, but taken captured using the GFP channel on the microscope. These files had the 
%_GFP.png file extension. Background images were obtained for each image by selecting a
%vesicle free region of the _GFP.png images and saving the region of interest with a
%_background.png file extension. The mean grayscale pixel intensity of the
%background image is substracted form the mean pixel intensity of the
%individual vesicles.

close all
clc
clear all
clearvars -except meanGFPIntensity1 meanGFPIntensity2 meanGFPIntensity3 meanGFPIntensity4 meanGFPIntensity5 meanGFPIntensity6 meanGFPIntensity7 meanGFPIntensity8 meanGFPIntensity9 meanGFPIntensity10

markerImage = imread('Image001_TXR.png'); %Microscopy image of vesicles containing dye label to visualise where all the vesicles are
detectionImage = imread('Image002_GFP.png'); %Microscopy image of vesicles containing the fluorescence species you wish to measure the intensity of

%Convert images to grayscale
grayMarkerImage = rgb2gray(markerImage);
grayDetectionImage = rgb2gray(detectionImage);

%Vesicle detection parameters
sensitivity_factor = 0.85; % Sensitivity ranges from 0 (Perfect circle) to 1 (Barely a circle).
min_diameter_micron = 5; % Minimum vesicle diameter in microns
max_diameter_micron = 20; % Maximal vesicle diameter in microns

%Converts the min and max diameters from micron units into pixel units,
%then assigns the radii length in pixels to variables a and b
min_diameter_pixel = min_diameter_micron*1920/124.5;
max_diameter_pixel = max_diameter_micron*1920/124.5;
a= round(min_diameter_pixel/2,0,'decimals');
b= round(max_diameter_pixel/2,0,'decimals');

%Uses a circular Hough transfer to find circle edges in grayscale images
[centers, radii] = imfindcircles(grayMarkerImage,[a b],'Sensitivity',sensitivity_factor);

%Visualise which vesicles are detected using the given parameters
figure('Renderer', 'painters', 'Position', [50 100 1400 600])
subplot(1,2,1)
imshow(markerImage)
viscircles(centers, radii,'EdgeColor','c');
text(50,-100,'Vesicles identified in marker image','Fontsize',18)

subplot(1,2,2)
imshow(detectionImage)
viscircles(centers, radii,'EdgeColor','c');
text(80,-100,'mNG expression in those vesicles','Fontsize',18)


%% Note
%After confirming vesicles ccan be detected by circular hough transform
%and suitable values for the sensitivity factor and diameter were 
%identified, the subsequent scripts were used to generate metric from all files
%coressponding to a single sample. Samples were groups within individual
%directories. The script is ran 1 sample at a time and the data stored
%within a meanGFPIntensity# corresponding with the sample number.

 
%% Step1 - Read in file names %%

close all
clc
clearvars -except meanGFPIntensity0 meanGFPIntensity2 meanGFPIntensity4 meanGFPIntensity6 meanGFPIntensity8 radii_array0 radii_array2 radii_array4 radii_array6 radii_array8

'Read files started'
files = dir('*.png'); %Select file format to read   
nfiles = length(files);  % Number of files found

GFP_counter = 0;
TXR_counter = 0;
Background_counter = 0;

%For loop iterates function for all files in the directory
for i=1:nfiles
    
    %Searches file names for handles
    pattern_TXR = 'TXR';
    pattern_GFP = 'GFP';
    pattern_Background = 'Background';
    
    current_filename = files(i).name;
    
    %Converts the file to grayscale and groups data into respective array
    if contains(current_filename,pattern_TXR)==1 %If the image is from the microscope's TXR channel
        TXR_counter = TXR_counter + 1;
        original_image = imread(current_filename);
        gray_image = rgb2gray(original_image);
        original_TXR_images{TXR_counter} = original_image;
        gray_TXR_images{TXR_counter} = gray_image;
        TXR_image_names{TXR_counter} = current_filename;
        
    elseif contains(current_filename,pattern_GFP)==1 %If the image is from the microscopes GFP channel
        GFP_counter = GFP_counter + 1;
        original_image = imread(current_filename);
        gray_image = rgb2gray(original_image);
        original_GFP_images{GFP_counter} = original_image;
        gray_GFP_images{GFP_counter} = gray_image;
        GFP_image_names{GFP_counter} = current_filename;
       
    elseif contains(current_filename,pattern_Background)==1 %If images is a sample of the background
        Background_counter = Background_counter + 1;
        original_image = imread(current_filename);
        gray_image = rgb2gray(original_image);
        original_background_images{Background_counter} = original_image;
        gray_background_images{Background_counter} = gray_image;
        background_image_names{Background_counter} = current_filename;
       
    else  
        ;
        
    end
end

'Read files done'

%% Step 2 - Detect vesicles and obtain metrics %%

'Vesicle detection started'
 
radii_counter = 1;

%Vesicle detection parameters
sensitivity_factor = 0.85; % Sensitivity ranges from 0 (Perfect circle) to 1 (Barely a circle).
min_diameter_micron = 5; % Minimum vesicle diameter in microns
max_diameter_micron = 20; % Maximal vesicle diameter in microns

%Converts the min and max diameters from micron units into pixel units,
%then assigns the radii length in pixels to variables a and b
min_diameter_pixel = min_diameter_micron*1920/124.5;
max_diameter_pixel = max_diameter_micron*1920/124.5;
a = round(min_diameter_pixel/2,0,'decimals');
b = round(max_diameter_pixel/2,0,'decimals');

%For each image in the directory, this loop reads the TXR channel image
%data and detects vesicles - the centers and radii of the circles are
%identified and stored. Logical masks are made corresponding with the dimensions of
%the original image, and the dimensions of an individual vesicle (via centers and radii) are used
%to create a single vesicle mask. This is applied to the GFP channel, and
%the pixel intensity read. This loops for every vesicle in each image, and
%for every image in the directory

for j=1:length(gray_TXR_images)
    
    [centers, radii] = imfindcircles(gray_TXR_images{j},[a b],'sensitivity',sensitivity_factor);

    [rows,columns] = size(gray_TXR_images{j});
    
    Background = mean(mean(gray_background_images{j}));
    meanVesicleIntensity = zeros(length(radii),1); %preassigned
    
        
    for k=1:numel(radii)
        
        %creates an array of logical zeros that matches the dimensions of the original image
        circleImage = false(rows, columns);
        [x, y] = meshgrid(1:columns, 1:rows);

        %Create a logical circle in the logical array
        circleImage((x - centers(k,1)).^2 + (y - centers(k,2)).^2 <= radii(k).^2) = true;

        % Mask the original image with the circle
        masked_TXR_image = gray_TXR_images{j}; %To measure TXR channel vesicle fluorescence
        masked_TXR_image(~circleImage) = 0;

        masked_GFP_image = gray_GFP_images{j}; %To measure GFP channel vesicle fluorescence
        masked_GFP_image(~circleImage) = 0;
        
        %Find the pixel intensity for this vesicle
        meanGFPIntensity1(radii_counter) = mean(masked_GFP_image(circleImage))-Background; %Rename variable for each sample (Directory)
        radii_array1(radii_counter) = radii(k);
        radii_counter = radii_counter + 1;
    end

end

'Vesicle detection finished'

%% Diameter vs mNG conversion

image_length = 124.5; %Lengths in um
image_pixels = 1920;%Length in pixels

um_per_pixel = image_length/image_pixels;

dimensions_array0 = 2.*radii_array0*um_per_pixel;
dimensions_array2 = 2.*radii_array2*um_per_pixel;
dimensions_array4 = 2.*radii_array4*um_per_pixel;
dimensions_array6 = 2.*radii_array6*um_per_pixel;
dimensions_array8 = 2.*radii_array8*um_per_pixel;

%% Plot GUV diamter vs mNG expression
figure('Renderer', 'painters', 'Position', [10 10 1600 600])
subplot(1,5,1)
size = 50;
s1 = scatter(dimensions_array0,meanGFPIntensity0,size)
s1.MarkerEdgeColor = [0 0 0]
s1.MarkerFaceColor = [0.4660 0.6740 0.1880];
s1.MarkerFaceAlpha = 0.4;
s1.LineWidth = 0.5;

text(1,87,'t=0','Fontsize',18)

ylim([-2 90])
yticks([0:20:80])
xlim([0 20])
xticks([0:5:20])
ylabel('Fluorescence (Gray value)');
xlabel('GUV diameter (\mum)');

ax1 = gca;
ax1.YAxis.FontSize = 18;
ax1.XAxis.FontSize = 18;
ax1.YAxis.Exponent = 0;                
ax1.LineWidth=2;
box off

subplot(1,5,2)
size = 50;
s1 = scatter(dimensions_array2,meanGFPIntensity2,size)
s1.MarkerEdgeColor = [0 0 0];
s1.MarkerFaceColor = [0.4660 0.6740 0.1880];
s1.MarkerFaceAlpha = 0.4;
s1.LineWidth = 0.5;

text(1,87,'t=2','Fontsize',18)

ylim([-2 90])
yticks([0:20:80])
xlim([0 20])
xticks([0:5:20])
ylabel('Fluorescence (Gray value)');
xlabel('GUV diameter (\mum)');

ax1 = gca;
ax1.YAxis.FontSize = 18;
ax1.XAxis.FontSize = 18;
ax1.YAxis.Exponent = 0;                
ax1.LineWidth=2;
box off

       
subplot(1,5,3)
size = 50;
s1 = scatter(dimensions_array4,meanGFPIntensity4,size)
s1.MarkerEdgeColor = [0 0 0];
s1.MarkerFaceColor = [0.4660 0.6740 0.1880];
s1.MarkerFaceAlpha = 0.4;
s1.LineWidth = 0.5;

text(1,87,'t=4','Fontsize',18)

ylim([-2 90])
yticks([0:20:80])
xlim([0 20])
xticks([0:5:20])
ylabel('Fluorescence (Gray value)');
xlabel('GUV diameter (\mum)');

ax1 = gca;
ax1.YAxis.FontSize = 18;
ax1.XAxis.FontSize = 18;
ax1.YAxis.Exponent = 0;                
ax1.LineWidth=2;
box off

subplot(1,5,4)
size = 50;
s1 = scatter(dimensions_array6,meanGFPIntensity6,size)
s1.MarkerEdgeColor = [0 0 0];
s1.MarkerFaceColor = [0.4660 0.6740 0.1880];
s1.MarkerFaceAlpha = 0.4;
s1.LineWidth = 0.5;

text(1,87,'t=6','Fontsize',18)

ylim([-2 90])
yticks([0:20:80])
xlim([0 20])
xticks([0:5:20])
ylabel('Fluorescence (Gray value)');
xlabel('GUV diameter (\mum)');

ax1 = gca;
ax1.YAxis.FontSize = 18;
ax1.XAxis.FontSize = 18;
ax1.YAxis.Exponent = 0;                
ax1.LineWidth=2;
box off

subplot(1,5,5)
size = 50;
s1 = scatter(dimensions_array8,meanGFPIntensity8,size)
s1.MarkerEdgeColor = [0 0 0];
s1.MarkerFaceColor = [0.4660 0.6740 0.1880];
s1.MarkerFaceAlpha = 0.4;
s1.LineWidth = 0.5;

text(1,87,'t=8','Fontsize',18)

ylim([-2 90])
yticks([0:20:80])
xlim([0 20])
xticks([0:5:20])
ylabel('Fluorescence (Gray value)');
xlabel('GUV diameter (\mum)');

ax1 = gca;
ax1.YAxis.FontSize = 18;
ax1.XAxis.FontSize = 18;
ax1.YAxis.Exponent = 0;                
ax1.LineWidth=2;
box off


%% Violin plot grouped by sample
figure2 = figure('Renderer', 'painters', 'Position', [10 10 500 700])


n=1.5;
al_goodplot(meanGFPIntensity0,1*n);
al_goodplot(meanGFPIntensity2,2*n);
al_goodplot(meanGFPIntensity4,3*n);
al_goodplot(meanGFPIntensity6,4*n);
al_goodplot(meanGFPIntensity8,5*n);

ylabel('GFP channel fluorescence (Gray value)')
ylim([0 110])

xlim([0 6*n])
xticks([n,2*n,3*n,4*n,5*n])
xticklabels({'t=0','t=2','t=4','t=6','t=8'});
xtickangle(30)

ax1 = gca;
ax1.YAxis.FontSize = 18;
ax1.XAxis.FontSize = 18;
ax1.YAxis.Exponent = 0;                
ax1.LineWidth=2;
box off

%% Violin plot function downloaded from https://www.mathworks.com/matlabcentral/fileexchange/91790-al_goodplot-boxblot-violin-plot

function [h, mu, sigma, q, notch] = al_goodplot(x, pos, boxw, col, type, bw, p)
% Violin and box plots for visualization of data distribution.
%
%   Inputs:
%     - x: NxP, data (P plots) (sample data if empty).
%     - pos: 1xP, position of the graphs in x-axis, default: 1.
%     - boxw: width of the graphs, default: 0.5.
%     - col: Px3 or 1x3, colors of the graphs. default: current color.
%     - type: laterality of the graph, 'left', 'right', 'bilateral' (default), or display manual: 'man'.
%     - bw: 1xP or 1x1, width of the window for kernel density. default: matlab default.
%     - p: increment for parzen (use the same p for 2 plots to be compared
%     to enforce the same area.). default: std/1000
%
%   Outputs:
%     - h: figure handle
%     - mu: mean
%     - sigma: standard deviation
%     - q: quantiles (0 1/4 1/2 3/4 1 1/10 9/10 1/100 99/100)
%     - notch: 95% confidence interval for median
% Parse inputs and set default values
if nargin<5 || isempty(type)
    type='bilateral';
end
if nargin<4 || isempty(col)
    colorOrder = get(gca, 'ColorOrder');
    col=colorOrder(mod(length(get(gca, 'Children')), size(colorOrder, 1))+1, :);
end
if nargin<3 || isempty(boxw)
    boxw=0.5;
end
if nargin<1 || isempty(x)
    type='man';
    % Example data for manual display
    rng(1)
    x=[4+randn(100,1); 8+3*randn(100,1)];
end
if nargin<2 || isempty(pos)
    pos=1:size(x,2);
end
if nargin<6 || isempty(p)  
    p=std(x(:))/1000;
end
u=0.9*min(x(:)):p:1.1*max(x(:));
h=cell(1,size(x,2));
mu=zeros(1,size(x,2));
sigma=zeros(1,size(x,2));
q=zeros(9,size(x,2));
notch=zeros(2,size(x,2));
if size(x,1)==1, x=x'; end
if size(x,2)>1 && size(pos,1)==1, pos=repmat(pos,1,size(x,2)); end
if size(x,2)>1 && size(col,1)==1, col=repmat(col,size(x,2),1); end
if size(x,2)>1 && size(boxw,1)==1, boxw=repmat(boxw,1,size(x,2)); end
for i=1:size(x,2)
    % Compute statistics useful to display
    mu(i)=mean(x(:,i));
    sigma(i)=std(x(:,i));
    q(:,i)=al_quantile(x(:,i),[0 1/4 1/2 3/4 1 1/10 9/10 1/100 99/100]);
    notch(:,i)=[q(3,i)-1.57*(q(4,i)-q(2,i))/sqrt(size(x,1)) q(3,i)+1.57*(q(4,i)-q(2,i))/sqrt(size(x,1))];
    
    % Compute kernel density
    uc=u(u>q(8,i) & u<q(9,i));
    if nargin<6 || isempty(bw)
        f=[0 al_parzen(x(:,i), uc) 0];
    else
        f=[0 al_parzen(x(:,i), uc, bw(i)) 0];
    end
    uc=[q(8,i) uc q(9,i)]; %#ok<AGROW>
    f=boxw(i)*2200*p*f/length(x);
    
    % Plots
    h{i}=gcf;
    switch type
        case {'bilateral', 'man'}
            scatter(pos(i)*ones(size(x(:,i))),x(:,i),10,col(i,:),'filled');
            hold on
            patch([pos(i)-f fliplr(pos(i)+f)], [uc fliplr(uc)], 0.97*col(i,:),'edgecolor','none','facealpha',0.15)
            patch([pos(i)+boxw(i)/2 pos(i)+boxw(i)/2 pos(i)+boxw(i)/4 pos(i)+boxw(i)/2 pos(i)+boxw(i)/2 pos(i)-boxw(i)/2 pos(i)-boxw(i)/2 pos(i)-boxw(i)/4 pos(i)-boxw(i)/2 pos(i)-boxw(i)/2], [q(2,i) notch(1,i) q(3,i) notch(2,i) q(4,i) q(4,i) notch(2,i) q(3,i) notch(1,i) q(2,i)], 0.97*col(i,:),'edgecolor','none','facealpha',0.5)
            patch([pos(i)-boxw(i)/8 pos(i)+boxw(i)/8 pos(i)+boxw(i)/8 pos(i)-boxw(i)/8 pos(i)-boxw(i)/8], [mu(i)-sigma(i) mu(i)-sigma(i) mu(i)+sigma(i) mu(i)+sigma(i) mu(i)-sigma(i)], col(i,:),'edgecolor','none','facealpha',0.35)
            plot([pos(i)-boxw(i)/4 pos(i)+boxw(i)/4], [q(3,i) q(3,i)],'color',col(i,:)/2,'linewidth',1)
            plot(pos(i), mu(i),'*','color',col(i,:)/2,'linewidth',1)
            
            if strcmp(type, 'man')
                % Display graph documentation
                pu=floor(mean(find(uc>q(4,i) & uc<q(7,i))));
                plot([pos(i)+f(pu),pos(i)+1.2*boxw(i)], [uc(pu), uc(pu)],':','color','k');
                text(pos(i)+1.2*boxw(i), uc(pu),' kernel density','clipping', 'on');
                plot([pos(i)+boxw(i)/2,pos(i)+1.2*boxw(i)], [notch(1,i), notch(1,i)],':','color','k')
                text(pos(i)+1.2*boxw(i), notch(1,i),' notch inf., 95% conf. median');
                plot([pos(i)+boxw(i)/2,pos(i)+1.2*boxw(i)], [notch(2,i), notch(2,i)],':','color','k')
                text(pos(i)+1.2*boxw(i), notch(2,i),' notch sup., 95% conf. median');
                plot([pos(i)+boxw(i)/2,pos(i)+1.2*boxw(i)], [q(2,i), q(2,i)],':','color','k')
                text(pos(i)+1.2*boxw(i), q(2,i),' 1st quartile, q(0.25)');
                plot([pos(i)+boxw(i)/2,pos(i)+1.2*boxw(i)], [q(4,i), q(4,i)],':','color','k')
                text(pos(i)+1.2*boxw(i), q(4,i),' 3rd quartile, q(0.75)');
                plot([pos(i)+boxw(i)/4,pos(i)+1.2*boxw(i)], [q(3,i), q(3,i)],':','color','k')
                text(pos(i)+1.2*boxw(i), q(3,i),' median, q(0.5)');
                plot([pos(i)+boxw(i)/8,pos(i)+1.2*boxw(i)], [mu(i)-sigma(i), mu(i)-sigma(i)],':','color','k')
                text(pos(i)+1.2*boxw(i), mu(i)-sigma(i),' mean - standard deviation');
                plot([pos(i)+boxw(i)/8,pos(i)+1.2*boxw(i)], [mu(i)+sigma(i), mu(i)+sigma(i)],':','color','k')
                text(pos(i)+1.2*boxw(i), mu(i)+sigma(i),' mean + standard deviation');
                plot([pos(i)+f(2),pos(i)+1.2*boxw(i)], [q(8,i), q(8,i)],':','color','k')
                text(pos(i)+1.2*boxw(i), q(8,i),' 1st percentile, q(0.01)');
                plot([pos(i)+f(length(f)-1),pos(i)+1.2*boxw(i)], [q(9,i), q(9,i)],':','color','k')
                text(pos(i)+1.2*boxw(i), q(9,i),' 99th percentile, q(0.99)');
                plot([pos(i),pos(i)+1.2*boxw(i)], [mu(i), mu(i)],':','color','k')
                text(pos(i)+1.2*boxw(i), mu(i),' mean');
                plot([pos(i),pos(i)+1.2*boxw(i)], [q(5,i), q(5,i)],':','color','k')
                text(pos(i)+1.2*boxw(i), q(5,i),' raw data');
                plot(pos(i)+3*boxw(i),0)
            end
            
        case 'left'
            scatter((pos(i)-boxw(i)/40)*ones(size(x(:,i))),x(:,i),10,col(i,:),'filled');
            hold on
            patch(pos(i)-f, uc, 0.97*col(i,:),'edgecolor','none','facealpha',0.15)
            patch([pos(i) pos(i)-boxw(i)/2 pos(i)-boxw(i)/2 pos(i)-boxw(i)/4 pos(i)-boxw(i)/2 pos(i)-boxw(i)/2 pos(i) pos(i)], [q(2,i) q(2,i) notch(1,i) q(3,i) notch(2,i) q(4,i) q(4,i) q(2,i)], 0.97*col(i,:),'edgecolor','none','facealpha',0.5)
            patch([pos(i)-boxw(i)/8 pos(i) pos(i) pos(i)-boxw(i)/8 pos(i)-boxw(i)/8], [mu(i)-sigma(i) mu(i)-sigma(i) mu(i)+sigma(i) mu(i)+sigma(i) mu(i)-sigma(i)], col(i,:),'edgecolor','none','facealpha',0.35)
            plot([pos(i)-boxw(i)/4 pos(i)], [q(3,i) q(3,i)],'color',col(i,:)/2,'linewidth',1.5)
            plot(pos(i)-boxw(i)/20, mu(i),'*','color',col(i,:)/2,'linewidth',1)
            
        case 'right'
            scatter((pos(i)+boxw(i)/40)*ones(size(x(:,i))),x(:,i),10,col(i,:),'filled');
            hold on
            patch(pos(i)+f, uc, 0.97*col(i,:),'edgecolor','none','facealpha',0.15)
            patch([pos(i) pos(i)+boxw(i)/2 pos(i)+boxw(i)/2 pos(i)+boxw(i)/4 pos(i)+boxw(i)/2 pos(i)+boxw(i)/2 pos(i) pos(i)], [q(2,i) q(2,i) notch(1,i) q(3,i) notch(2,i) q(4,i) q(4,i) q(2,i)], 0.97*col(i,:),'edgecolor','none','facealpha',0.5)
            patch([pos(i)+boxw(i)/8 pos(i) pos(i) pos(i)+boxw(i)/8 pos(i)+boxw(i)/8], [mu(i)-sigma(i) mu(i)-sigma(i) mu(i)+sigma(i) mu(i)+sigma(i) mu(i)-sigma(i)], col(i,:),'edgecolor','none','facealpha',0.35)
            plot([pos(i)+boxw(i)/4 pos(i)], [q(3,i) q(3,i)],'color',col(i,:)/2,'linewidth',1.5)
            plot(pos(i)+boxw(i)/20, mu(i),'*','color',col(i,:)/2,'linewidth',1)
            
    end
end
grid on
box on
end
% Stat functions to avoid using the statistical toolbox
function q = al_quantile(x, p)
sx=sort(x);
indx=(length(x)-1)*p+1;
q=zeros(1,length(p));
for i=1:length(p)
    if floor(indx(i))==indx(i)
        q(i)=sx(indx(i));
    else
        q(i)=(sx(floor(indx(i)))+sx(floor(indx(i))+1))/2;
    end
end
end
function f = al_parzen(x, u, bw)
q=al_quantile(x,[1/4 3/4]);
if nargin<3 || isempty(bw)
    bw=0.9*min(std(x),(q(2)-q(1))/1.35)*length(x)^(-1/5); % Silverman's rule of thumb
end
f=zeros(size(u));
for i=1:length(x)
    k=(1/(bw*sqrt(2*pi)))*exp(-0.5*((x(i)-u)/bw).^2);
    f=f+k;
end
end