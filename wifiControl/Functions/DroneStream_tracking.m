%% DroneStream_Tracking.m
%  Real-time AR.Drone stream with adaptive Kalman Filter tracking
%  Combines DroneStream.m (double-buffer MJPEG reader) with
%  tracking_KF_adapt_aceleration.m (YOLO + KF pipeline)
%
%  Colour code of the bounding box:
%    Yellow  -> confirming target
%    Red     -> tracking (KF + YOLO)
%    Green   -> KF-only prediction (YOLO temporarily lost)

%% ── 1. Configuration ────────────────────────────────────────────────────

% --- Stream files (written by drone_stream.py) ---
jpgA     = '/tmp/droneA.jpg';
jpgB     = '/tmp/droneB.jpg';
slotFile = '/tmp/drone_slot.txt';

% --- ONNX models ---
modelPathDetection = '/Users/usuario/Desktop/MEAer/TFM/Software/ONXX Models/best_640_16_downsampled.onnx';
modelPathTracking  = '/Users/usuario/Desktop/MEAer/TFM/Software/ONXX Models/best_320_16_downsampled.onnx';

% --- Network input sizes ---
imageSizeDetection = 640;
imageSizeTracking  = 320;
inputShapeDetection = [imageSizeDetection, imageSizeDetection];
inputShapeTracking  = [imageSizeTracking,  imageSizeTracking];

% --- Assumed stream frame rate (used for adaptive-Q dt) ---
STREAM_FPS = 30;
dt = 1 / STREAM_FPS;

% --- FPS display: ring buffer ---
VENTANA    = 30;
tHistorial = nan(1, VENTANA);

%% ── 2. Launch drone_stream.py in background ─────────────────────────────

pyExe    = '/opt/anaconda3/envs/drone_env/bin/python3';
pyScript = '/Users/usuario/Desktop/MEAer/TFM/Software/Real Time Tracking/drone_stream.py';
system([pyExe ' "' pyScript '" &']);
fprintf('Python launched in background...\n');

%% ── 3. Load YOLO models in parallel workers ─────────────────────────────

addpath(genpath(pwd));

pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('local', 2);
end

futureInit1 = parfeval(pool, @yoloTrackingWorker, 3, [], modelPathDetection, inputShapeDetection, 'detection');
fetchOutputs(futureInit1);
fprintf('Detection model ready\n')

futureInit2 = parfeval(pool, @yoloTrackingWorker, 3, [], modelPathTracking, inputShapeTracking, 'tracking');
fetchOutputs(futureInit2);
fprintf('Tracking model ready\n\n')

%% ── 4. Wait for first frame ─────────────────────────────────────────────

fprintf('Waiting for first frame...\n');
while ~exist(slotFile, 'file')
    pause(0.1);
end
pause(0.5);

%% ── 5. Kalman Filter state initialisation ───────────────────────────────

kf              = [];
kfInitialized   = false;
bboxes          = [];
lastBboxes      = [];
lastPredBbox    = [];
futureInference = [];
isTracking      = false;
isConfirming    = false;
missCount       = 0;
maxMisses       = 5;
predFrameCount  = 0;
maxPredFrames   = 10;
confirmCount    = 0;
minConfirmCount = 2;
lastCenter      = [];
adaptiveQ       = 200;

sentX1 = 1; sentY1 = 1; sentScaleX = 1; sentScaleY = 1;
inferenceStartTime  = [];
isTrackingInference = false;

%% ── 6. Display window ───────────────────────────────────────────────────

hFig = figure('Name', 'AR.Drone – KF Tracking');
hImg = imshow(zeros(360, 640, 3, 'uint8'));
title('Waiting for stream...');
fprintf('Display started. Close the figure to stop.\n\n');

%% ── 7. Main loop ────────────────────────────────────────────────────────

tGlobal      = tic;
tPrimerFrame = nan;
frameCount   = 0;
ultimoSlot   = '';

try
    while ishandle(hFig)

        % ── 7.1  Read next frame from double-buffer ───────────────────
        frame = [];
        if exist(slotFile, 'file')
            fid  = fopen(slotFile, 'r');
            slot = fgetl(fid);
            fclose(fid);

            if ~strcmp(slot, ultimoSlot)
                jpgRead = jpgA;
                if strcmp(slot, 'B'), jpgRead = jpgB; end

                try
                    imgRaw = imread(jpgRead);
                    if ~isempty(imgRaw)
                        frame      = imgRaw;
                        ultimoSlot = slot;
                        frameCount = frameCount + 1;
                        tAhora     = toc(tGlobal);

                        if frameCount == 1
                            tPrimerFrame = tAhora;
                        end

                        % Rolling FPS
                        idx      = mod(frameCount - 1, VENTANA) + 1;
                        tAntiguo = tHistorial(idx);
                        tHistorial(idx) = tAhora;
                        if frameCount >= VENTANA && ~isnan(tAntiguo)
                            dt_fps = tAhora - tAntiguo;
                            if dt_fps > 0
                                fprintf('Stream FPS (30-frame avg): %.2f\n', VENTANA / dt_fps);
                            end
                        end
                    end
                catch
                    % Frame not ready yet – skip tracking this iteration
                end
            end
        end

        % Nothing new from the stream – keep the display alive and wait
        if isempty(frame)
            drawnow limitrate;
            continue;
        end

        % ── 7.2  Kalman prediction (tracking mode, inference pending) ─
        if isTracking && ~isConfirming && kfInitialized && ~isempty(lastBboxes) && ~isempty(futureInference)
            kfPred = predict(kf);
            w = lastBboxes(3); h = lastBboxes(4);
            bboxes = [kfPred(1)-w/2, kfPred(3)-h/2, w, h];
        end

        % ── 7.3  Kalman prediction (detection mode, object temporarily lost) ─
        if ~isTracking && kfInitialized && ~isempty(lastBboxes) && ~isempty(futureInference)
            if predFrameCount < maxPredFrames
                kfPred         = predict(kf);
                w = lastBboxes(3); h = lastBboxes(4);
                bboxes         = [kfPred(1)-w/2, kfPred(3)-h/2, w, h];
                lastPredBbox   = bboxes;
                predFrameCount = predFrameCount + 1;
            else
                bboxes = lastPredBbox;
            end
        end

        % ── 7.4  Launch inference if no task is active ────────────────
        if isempty(futureInference)

            if isTracking && ~isempty(lastBboxes)
                % --- Tracking mode: crop around KF prediction ---
                if kfInitialized && ~isConfirming
                    state    = kf.State;
                    xc = state(1); yc = state(3);
                    xcLast   = lastBboxes(1) + lastBboxes(3)/2;
                    ycLast   = lastBboxes(2) + lastBboxes(4)/2;
                    maxDrift = imageSizeTracking * 0.5;
                    xc = xcLast + max(-maxDrift, min(maxDrift, xc - xcLast));
                    yc = ycLast + max(-maxDrift, min(maxDrift, yc - ycLast));
                else
                    xc = lastBboxes(1) + lastBboxes(3)/2;
                    yc = lastBboxes(2) + lastBboxes(4)/2;
                end

                x1 = round(xc - imageSizeTracking/2);
                y1 = round(yc - imageSizeTracking/2);
                x2 = x1 + imageSizeTracking - 1;
                y2 = y1 + imageSizeTracking - 1;

                x1 = max(1, x1); y1 = max(1, y1);
                x2 = min(x2, size(frame,2));
                y2 = min(y2, size(frame,1));

                cropRaw = frame(y1:y2, x1:x2, :);
                cropH = y2-y1+1; cropW = x2-x1+1;

                if cropH ~= imageSizeTracking || cropW ~= imageSizeTracking
                    crop   = imresize(cropRaw, [imageSizeTracking, imageSizeTracking]);
                    scaleX = cropW / imageSizeTracking;
                    scaleY = cropH / imageSizeTracking;
                else
                    crop   = cropRaw;
                    scaleX = 1; scaleY = 1;
                end

                sentX1 = x1; sentY1 = y1;
                sentScaleX = scaleX; sentScaleY = scaleY;
                futureInference     = parfeval(pool, @yoloTrackingWorker, 3, crop, modelPathTracking, inputShapeTracking, 'tracking');
                inferenceStartTime  = tic;
                isTrackingInference = true;

            else
                % --- Detection mode: full frame ---
                futureInference     = parfeval(pool, @yoloTrackingWorker, 3, frame, modelPathDetection, inputShapeDetection, 'detection');
                inferenceStartTime  = tic;
                isTrackingInference = false;
            end
        end

        % ── 7.5  Collect inference results when ready ─────────────────
        if ~isempty(futureInference) && strcmp(futureInference.State, 'finished')
            try
                [bboxesTmp, ~, ~] = fetchOutputs(futureInference);
            catch
                bboxesTmp = [];
            end

            if isTrackingInference && ~isempty(inferenceStartTime)
                fprintf('Tracking inference: %.1f ms\n', toc(inferenceStartTime)*1e3);
            end
            inferenceStartTime  = [];
            isTrackingInference = false;
            futureInference     = [];

            if isTracking
                % ── Tracking branch ──────────────────────────────────
                if ~isempty(bboxesTmp)
                    % Map crop coords → full-frame coords
                    bboxes      = bboxesTmp;
                    bboxes(:,1) = bboxesTmp(:,1) * sentScaleX + sentX1 - 1;
                    bboxes(:,2) = bboxesTmp(:,2) * sentScaleY + sentY1 - 1;
                    bboxes(:,3) = bboxesTmp(:,3) * sentScaleX;
                    bboxes(:,4) = bboxesTmp(:,4) * sentScaleY;
                    lastBboxes  = bboxes(1,:);

                    xc = lastBboxes(1) + lastBboxes(3)/2;
                    yc = lastBboxes(2) + lastBboxes(4)/2;

                    if isConfirming
                        confirmCount = confirmCount + 1;
                        if confirmCount >= minConfirmCount
                            disp('Target confirmed. Starting tracking...')
                            isConfirming = false;
                            kf = trackingKF('MotionModel','2D Constant Velocity', ...
                                            'State',            [xc;0;yc;0], ...
                                            'MeasurementNoise', 10, ...
                                            'ProcessNoise',     adaptiveQ);
                            kfInitialized = true;
                            lastCenter    = [xc, yc];
                        end
                    else
                        if ~kfInitialized
                            kf = trackingKF('MotionModel','2D Constant Velocity', ...
                                            'State',            [xc;0;yc;0], ...
                                            'MeasurementNoise', 10, ...
                                            'ProcessNoise',     adaptiveQ);
                            kfInitialized = true;
                            lastCenter    = [xc, yc];
                        else
                            % Adaptive process noise based on estimated acceleration
                            if ~isempty(lastCenter)
                                accel = norm([xc, yc] - lastCenter) / dt;
                                if accel > 70
                                    adaptiveQ           = 300;
                                    kf.MeasurementNoise = 5;   % trust YOLO more
                                elseif accel > 15
                                    adaptiveQ           = 200;
                                    kf.MeasurementNoise = 10;  % normal
                                else
                                    adaptiveQ           = 50;
                                    kf.MeasurementNoise = 20;  % smooth at low speed
                                end
                                kf.ProcessNoise = adaptiveQ;
                            end
                            lastCenter = [xc, yc];
                            correct(kf, [xc, yc]);
                        end
                    end

                    missCount      = 0;
                    predFrameCount = 0;
                    lastPredBbox   = [];

                    if ~isConfirming
                        state  = kf.State;
                        bboxes = [state(1)-lastBboxes(3)/2, state(3)-lastBboxes(4)/2, lastBboxes(3), lastBboxes(4)];
                    end

                else
                    % No detection in tracking mode
                    missCount = missCount + 1;

                    if isConfirming
                        disp('Confirmation failed. Back to detection...')
                        isTracking   = false; isConfirming = false;
                        confirmCount = 0; missCount = 0;
                        lastBboxes   = []; bboxes = []; lastCenter = [];
                    else
                        disp(['Target not detected. Attempt ' num2str(missCount) '/' num2str(maxMisses)])
                        if missCount >= maxMisses
                            disp('Target lost. Switching to detection...')
                            isTracking     = false;
                            missCount      = 0; predFrameCount = 0; confirmCount = 0;
                            lastCenter     = []; adaptiveQ = 200;
                        end
                    end
                end

            else
                % ── Detection branch ─────────────────────────────────
                if ~isempty(bboxesTmp)
                    xcDet = bboxesTmp(1,1) + bboxesTmp(1,3)/2;
                    ycDet = bboxesTmp(1,2) + bboxesTmp(1,4)/2;

                    % Reject detections too far from last known position
                    isValid = true;
                    if ~isempty(lastBboxes)
                        xcLast = lastBboxes(1) + lastBboxes(3)/2;
                        ycLast = lastBboxes(2) + lastBboxes(4)/2;
                        dist   = norm([xcDet-xcLast, ycDet-ycLast]);
                        if dist > imageSizeTracking * 2.0
                            disp('Warning: false positive rejected')
                            isValid      = false;
                            confirmCount = 0;
                            bboxes       = lastPredBbox;
                        end
                    end

                    if isValid
                        disp('Target detected. Waiting for confirmation...')
                        lastBboxes     = bboxesTmp(1,:);
                        bboxes         = bboxesTmp;
                        isTracking     = true;
                        isConfirming   = true;
                        confirmCount   = 1;
                        missCount      = 0; predFrameCount = 0;
                        lastPredBbox   = [];
                    end
                else
                    confirmCount = 0;
                    if kfInitialized
                        missCount = missCount + 1;
                        if missCount >= maxMisses * 3
                            disp('No target found. Resetting tracker...')
                            kfInitialized = false;
                            lastBboxes    = []; lastPredBbox = [];
                            bboxes        = []; missCount = 0;
                            predFrameCount = 0; lastCenter = []; adaptiveQ = 200;
                        end
                    end
                end
            end
        end

        % ── 7.6  Annotate and display frame ──────────────────────────
        if ~isempty(bboxes)
            if isConfirming
                color = 'yellow';                          % waiting for confirmation
            elseif ~isTracking && predFrameCount >= maxPredFrames
                color = 'green';                           % pure KF prediction
            else
                color = 'red';                             % active tracking
            end
            frameDisplay = insertShape(frame, 'Rectangle', bboxes, 'Color', color, 'LineWidth', 2);
        else
            frameDisplay = frame;
        end

        set(hImg, 'CData', frameDisplay);
        drawnow limitrate;

    end  % while ishandle

catch e
    fprintf('Error: %s\n', e.message);
end

%% ── 8. Cleanup ──────────────────────────────────────────────────────────

tFinal = toc(tGlobal);
system('pkill -f drone_stream.py');
fprintf('Python stopped.\n');

if ~isnan(tPrimerFrame) && frameCount > 0
    tStream = tFinal - tPrimerFrame;
    fprintf('Total average FPS: %.2f  (%d frames / %.1f s)\n', ...
            frameCount / max(tStream, 0.001), frameCount, tStream);
else
    disp('No frames were processed.')
end