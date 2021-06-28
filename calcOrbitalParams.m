 function [sf, IRdown, visScattered, flatVis, flatIR, sky, CO2FrostPoints, nStepsInYear, lsWrapped, hr, ltst] = calcOrbitalParams2(slope, slope_aspect, ecc, obl, Lsp, s, dt, flatVisSaved, flatIRSaved)
%CALCORBITALPARAMS2 Summary of this function goes here
%   Detailed explanation goes here

% October 30, 2018: 1/2 factor in front for scattered vis added to account for half of scattered light 
% goes towards the ground, while half the scattered light goes towards
% space, as is in the Aharonson and Schorghofer paper
% October 30, 2018: Add in extinction due to atmospheric attenuation, as is in Schorghofer
% and Edgett 2006 and Aharonson and Schorghofer 2006

% Set constants
au  = 1.4959787061e11;           % Size of one astronomical unit in meters
gc  = 6.67259e-11;               % Gravitational constant
sm  = 1.9891e30;                 % Solar mass
yearLength = 2*pi/sqrt(gc*sm/(s.a*au)^3);   % Length of Mars year in seconds using Kepler's 3rd law
EarthYearLength = 2*pi*sqrt(au^3/(gc*sm));   % Length of one Earth year in seconds
f1au = 1367;                     % Solar Flux at 1AU (W/m^2)
d2r = pi/180;                    % For converting degrees to radians
r2d = 180/pi;                    % For converting radians to degrees

%% Describe orbit of a body
% Get solar distance and subsolar latitude for given time taking into
% account eccentricity, semi-major axis, obliquity and longitude of
% perihelion. Then convert to solar flux.

%clear TA MA EA t t2 sf IRdown visScattered sky flatVis flatIR

% Set slope and latitude values in radians
sloperad = deg2rad(slope);
slopeaspectrad = deg2rad(slope_aspect);
latrad = deg2rad(s.lat);

% Low-res calculation of times at uniformly spaced true anomalies
nn=1e4;
TA = ((0:nn)'+0.5)*2*pi/nn;        % create evenly spaced true anomalies
TA=TA(1:end-1);   % do because matlab indexes start at 1
EA = acos( (ecc+cos(TA))./(1+ecc*cos(TA)) );   % Calculate eccentric anomalies
MA = EA - ecc*sin(EA);                   % Calculate mean anomalies
t  = MA/sqrt(gc*sm/(s.a*au)^3);     %Time along Mars orbital path (will be irregularly spaced)
t(find(TA > pi)) = yearLength - t(find(TA > pi));  % correction of 2nd half of the orbit

% High-Res interpolation of true anomalies at uniformly spaced times
numTimesteps = ceil(yearLength/dt);
timesBeginAndEnd = [dt/2, numTimesteps*dt - (dt/2)];
lmstBeginAndEnd = ((mod((timesBeginAndEnd ./ s.rot .* 2*pi), (2*pi)) - pi)*r2d/15) + 12;  % Local mean solar time from 0 to 24 hours

% Make number of timesteps connect back to midnight
if lmstBeginAndEnd(end) - lmstBeginAndEnd(1) > 12
    numTimesteps = numTimesteps + round((mod(lmstBeginAndEnd(1)-lmstBeginAndEnd(end) + 24,24)/24.0 * s.rot/dt));
end
if lmstBeginAndEnd(end) - lmstBeginAndEnd(1) < 12
    numTimesteps = numTimesteps - round((mod(lmstBeginAndEnd(end)-lmstBeginAndEnd(1) + 24,24)/24.0 * s.rot/dt));
end

% Redistribute with new number of timesteps
t2 = ([0:numTimesteps-2] + 0.5) * dt; t2=t2'; % Make column vector
nStepsInYear = size(t2,1);

% Now use these evenly spaced times to calculate the rest
TA2 = interp1(t,TA,t2, 'pchip'); % Interpolate the True Anomalies at these times
EA2 = acos( (ecc+cos(TA2))./(1+ecc*cos(TA2)) );  % Calculate the Eccentric Anomalies at these times

% Solar distance/declination/longitude and local time calculations
sol_dist = s.a*(1 - ecc*cos(EA2));               % Solar Distance
lsrad = mod((TA2 + Lsp), (2*pi));  % in radians for taking sin of
ls = radtodeg(TA2 + Lsp); % so it wont wrap around in plots
lsWrapped = radtodeg(mod((TA2 + Lsp), (2*pi)));   % Solar longitude in degrees
sin_dec = sin(obl)*sin(lsrad);               % Sine of solar declination
cos_dec = sqrt(1 - sin_dec.^2);

% New redistributed Ls for plotting
lsNew = ls * 0; % Create array for new adjusted Ls values
lsOvershoot = ls(nStepsInYear) - ls(ceil(yearLength/dt)); % Ls at end of year in Ls time minus Ls at the end of the array with extra timesteps added
for ii = 1:nStepsInYear
    lsNew(ii) = ls(ii) - lsOvershoot*(ii/nStepsInYear);
end
lsWrapped = mod(lsNew, 360); % Ls values between 0 and 360
whereCrossOver360to0 = find(floor(lsWrapped)==0,1); % First index after Ls wrapped from 360 back to 0

hr = mod((t2/s.rot * 2*pi), (2*pi)) - pi;  % Local mean true solar time in hour angle (-pi to pi)- cant be used to compare to ephemeris time because fixed rotation...calculated as if was a mean solar
ltst = hr*r2d/15 + 12;  % Local time between 0 (midnight) to 12 (noon) and back to midnight at 24

% Incidence angles and fluxes
cosi = ( sin(latrad)*sin_dec + cos(latrad)*cos_dec.*cos(hr) );
cosi(cosi < -1.0) = -1.0; % Makes sure doesn't go outside [-1,1]
cosi(cosi > 1.0) = 1.0;

% Slopes
sini   = sqrt(1-cosi.^2);

% Value that goes into calculating solar azimuth- want to make sure it
% stays within [-1,1] or will get imaginary numbers
cos_az = (sin_dec - sin(latrad).*cosi) ./ (cos(latrad).*sini);
cos_az(cos_az > 1.0) = 1.0;
cos_az(cos_az < -1.0) = -1.0;
az  = acos(cos_az);
az(find(ltst > 12)) = 2*pi - az(find(ltst > 12)); % flip around since arccos degenerate

cosi_slope = cosi.*cos(sloperad) + sini.*sin(sloperad).*cos(slopeaspectrad - az);
cosi_slope(find(cosi_slope < 0)) = 0;  % Set any value less than 0 (shadowed) to 0
cosi_slope(find(cosi < 0)) = 0; 

% Solar Flux
sf = f1au./(sol_dist.^2) .* cosi_slope;

% No slope
if slope == 0
    cosi(find(cosi < 0)) = 0; 
    sf = f1au./(sol_dist.^2) .* cosi;
end

% Add in extinction due to atmospheric attenuation, as is in Schorghofer
% and Edgett 2006 and Aharonson and Schorghofer 2006
if slope == 0
    maxCoefAtmosAtt = max(sin(pi/2 - acos(cosi)), 0.04);
else
    maxCoefAtmosAtt = max(sin(pi/2 - acos(cosi_slope)), 0.04);
end
atmosAtt = (1-s.scatteredVisPerc-s.downwellingPerc).^(1./maxCoefAtmosAtt);
sfTOA = sf;
sf = sf.*atmosAtt;


annual_sf = sum(sf*dt);   % Total annual energy
fprintf('Solar Flux Min = %8.4f, Max = %8.4f and Mean = %8.4f [W/m^2]\n', min(sf), max(sf), mean(sf));
fprintf ('Total Annual Solar Flux = %.6e [W/m^2] \n', annual_sf);

% Daily noontime flux and the value of this to be used for downwelling IR
% radiation. Gets modified for sloped cases below.
sf_noon = f1au./(sol_dist.^2) .* ( sin(latrad)*sin_dec + cos(latrad)*cos_dec);
IRdown = s.downwellingPerc .* sf_noon;

% Scattered Visible Light at each timestep (will get 0 scattered vis light
% at night so use the sf array).
% This approximation assumes light scatters isotropically even though in
% actuality more light is scattered near the disk of the sun in the sky
% 1/2 factor in front for half scattered towards ground, half scattered
% towards space
visScattered = 0.5 .* s.scatteredVisPerc .* sfTOA;

% If sloped and the parameter file is set to first calculate reradiation
% from nearby terrain based on flat values, then load in those values.
if slope ~= 0 && s.slope_rerad == 1
    flatVis = flatVisSaved;
    flatIR = flatIRSaved;
    sky = cos(sloperad/2)^2;
else
    flatVis = zeros(nStepsInYear,1);
    flatIR = zeros(nStepsInYear,1);
    sky = 1;
end

% Calculate frost point temperatures vs time for a given elevation
atmPressTerms = [7.97078 -0.539781 0.468818 0.368771 -0.392702 0.0206071 -0.0224410 -0.0326866 -0.00261966 0.0145776 -0.000184519];
atmPress = atmPressTerms(1);
for ii=1:5
    atmPress = atmPress + atmPressTerms(ii*2)*sin(ii*lsrad) + atmPressTerms(ii*2+1)*cos(ii*lsrad); % inside the sin/cos could also do 2pi*ii*ls/360 if ls was in degrees
end

atmPress = atmPress*exp(-(s.elevation+3627)/s.atmScaleHeight); % Scale pressure from Viking Landing site at -3627 m elevation
CO2FrostPoints = 3148./(23.102 - log(atmPress)); % Sublimation point of CO2 ice
CO2FrostPoints = CO2FrostPoints .* s.frostSwitch; % Turns frost off by setting frost point to 0 if inputted in param file
end

