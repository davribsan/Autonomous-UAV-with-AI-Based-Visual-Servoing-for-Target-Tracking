% DroneStream_tracking_init.m
%  InitFcn callback for the ARDroneTT Simulink model.
%  Runs once at simulation start: launches Python, loads YOLO models,
%  waits for the first frame, and pushes all shared state to the base workspace.

%% ── 1. Configuration ────────────────────────────────────────────────────────

% Double-buffer stream files (written by drone_stream.py)
jpgA     = '/tmp/droneA.jpg';
jpgB     = '/tmp/droneB.jpg';
slotFile = '/tmp/drone_slot.txt';

% ONNX model paths
modelPathDetection = '/Users/usuario/Desktop/MEAer/TFM/End-end system/ARDroneSimulinkDevKit_V1.1 hover/wifiControl/ONXX Models/best_640_downsampled_2.onnx';
modelPathTracking  = '/Users/usuario/Desktop/MEAer/TFM/End-end system/ARDroneSimulinkDevKit_V1.1 hover/wifiControl/ONXX Models/best_320_downsampled_2.onnx';

% Network input sizes
imageSizeDetection  = 640;
imageSizeTracking   = 320;
inputShapeDetection = [imageSizeDetection, imageSizeDetection];
inputShapeTracking  = [imageSizeTracking,  imageSizeTracking];

%% ── 2. Launch drone_stream.py in background ─────────────────────────────────

pyExe    = '/opt/anaconda3/envs/drone_env/bin/python3';
pyScript = '/Users/usuario/Desktop/MEAer/TFM/End-end system/ARDroneSimulinkDevKit_V1.1 hover/wifiControl/drone_stream.py';
system([pyExe ' "' pyScript '" &']);
fprintf('Python launched in background...\n');

%% ── 3. Load YOLO models in parallel workers ─────────────────────────────────

% Add current folder and subfolders to path (main worker)
addpath(genpath(pwd));

% Start pool FIRST (pctRunOnAll needs it to exist)
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', 2);
end

% Now propagate the full path to all parallel workers
workerPath = '/Users/usuario/Desktop/MEAer/TFM/End-end system/ARDroneSimulinkDevKit_V1.1 hover/wifiControl/Functions';
pctRunOnAll addpath(genpath(workerPath));

% Load detection model and wait (30 s timeout)
futureInit1 = parfeval(pool, @yoloTrackingWorker, 3, [], modelPathDetection, inputShapeDetection, 'detection');
ok1 = wait(futureInit1, 'finished', 300);
if ~ok1
    error('Detection model failed to load within 30 s. Error: %s', futureInit1.Error.message);
end
fetchOutputs(futureInit1);
fprintf('Detection model ready\n')

% Load tracking model and wait (30 s timeout)
futureInit2 = parfeval(pool, @yoloTrackingWorker, 3, [], modelPathTracking, inputShapeTracking, 'tracking');
ok2 = wait(futureInit2, 'finished', 300);
if ~ok2
    error('Tracking model failed to load within 30 s. Error: %s', futureInit2.Error.message);
end
fetchOutputs(futureInit2);
fprintf('Tracking model ready\n\n')

%% ── 4. Wait for first frame ─────────────────────────────────────────────────

fprintf('Waiting for first frame...\n');
while ~exist(slotFile, 'file')
    pause(0.1);
end
pause(0.5);
fprintf('First frame received.\n');

%% ── 5. Kalman Filter state initialisation ───────────────────────────────────

kf                  = [];
kfInitialized       = false;
isTracking          = false;
isConfirming        = false;
missCount           = 0;
confirmCount        = 0;
predFrameCount      = 0;
lastBboxes          = [];
lastPredBbox        = [];
futureInference     = [];
lastCenter          = [];
adaptiveQ           = 200;
sentX1              = 1;
sentY1              = 1;
sentScaleX          = 1;
sentScaleY          = 1;
inferenceStartTime  = [];
isTrackingInference = false;

%% ── 6. Push all variables to the base workspace ─────────────────────────────
%  All assignin calls are here, AFTER every variable has been defined.

assignin('base', 'jpgA',                jpgA);
assignin('base', 'jpgB',                jpgB);
assignin('base', 'slotFile',            slotFile);
assignin('base', 'modelPathDetection',  modelPathDetection);
assignin('base', 'modelPathTracking',   modelPathTracking);
assignin('base', 'inputShapeDetection', inputShapeDetection);
assignin('base', 'inputShapeTracking',  inputShapeTracking);
assignin('base', 'pool',                pool);
assignin('base', 'kf',                  kf);
assignin('base', 'kfInitialized',       kfInitialized);
assignin('base', 'isTracking',          isTracking);
assignin('base', 'isConfirming',        isConfirming);
assignin('base', 'missCount',           missCount);
assignin('base', 'confirmCount',        confirmCount);
assignin('base', 'predFrameCount',      predFrameCount);
assignin('base', 'lastBboxes',          lastBboxes);
assignin('base', 'lastPredBbox',        lastPredBbox);
assignin('base', 'futureInference',     futureInference);
assignin('base', 'lastCenter',          lastCenter);
assignin('base', 'adaptiveQ',           adaptiveQ);
assignin('base', 'sentX1',             sentX1);
assignin('base', 'sentY1',             sentY1);
assignin('base', 'sentScaleX',         sentScaleX);
assignin('base', 'sentScaleY',         sentScaleY);
assignin('base', 'inferenceStartTime',  inferenceStartTime);
assignin('base', 'isTrackingInference', isTrackingInference);

fprintf('Base workspace initialised. Simulation ready.\n');