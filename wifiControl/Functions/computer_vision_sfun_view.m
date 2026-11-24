function computer_vision_sfun_view(block)
% COMPUTER_VISION_SFUN  Level-2 MATLAB S-Function for drone tracking.
%
% FIX: block.UserData not supported in Level-2 S-Functions.
%      A single persistent variable at file scope is used instead.
%      All subfunctions access it via the top-level mex-function closure.

setup(block);
end

%% =========================================================================
function setup(block)

    block.NumInputPorts  = 0;
    block.NumOutputPorts = 3;
    block.AllowSignalsWithMoreThan2D = true;

    % Output 1: s [1x2]
    block.OutputPort(1).Dimensions  = [1 2];
    block.OutputPort(1).DatatypeID  = 0;
    block.OutputPort(1).Complexity  = 'Real';

    % Output 2: bw (scalar)
    block.OutputPort(2).Dimensions  = 1;
    block.OutputPort(2).DatatypeID  = 0;
    block.OutputPort(2).Complexity  = 'Real';

    % Output 3: frameOut [360x640x3]
    block.OutputPort(3).Dimensions  = [360 640 3];
    block.OutputPort(3).DimensionsMode = 'Fixed';
    block.OutputPort(3).DatatypeID  = 3;
    block.OutputPort(3).Complexity  = 'Real';

    block.SampleTimes = [0.03 0];

    block.RegBlockMethod('Start',    @Start);
    block.RegBlockMethod('Outputs',  @Outputs);
    block.RegBlockMethod('Terminate',@Terminate);
end

%% =========================================================================
function Start(block) %#ok<INUSD>
% Initialise state in the base-workspace persistent container

    ud = struct();

    ud.pool                = evalin('base','pool');
    ud.modelPathDetection  = evalin('base','modelPathDetection');
    ud.modelPathTracking   = evalin('base','modelPathTracking');
    ud.inputShapeDetection = evalin('base','inputShapeDetection');
    ud.inputShapeTracking  = evalin('base','inputShapeTracking');
    ud.jpgA                = evalin('base','jpgA');
    ud.jpgB                = evalin('base','jpgB');
    ud.slotFile            = evalin('base','slotFile');

    % Tracker state
    ud.kf                  = [];
    ud.kfInitialized       = false;
    ud.isTracking          = false;
    ud.isConfirming        = false;
    ud.missCount           = 0;
    ud.confirmCount        = 0;
    ud.predFrameCount      = 0;
    ud.lastBboxes          = [];
    ud.lastCenter          = [];
    ud.lastPredBbox        = [];
    ud.lastRejectedCenter  = [];
    ud.confirmCenter       = [];   % position of first confirmation attempt
    ud.futureInference     = [];
    ud.adaptiveQ           = 200;

    % Crop-to-frame mapping
    ud.sentX1              = 1;
    ud.sentY1              = 1;
    ud.sentScaleX          = 1;
    ud.sentScaleY          = 1;

    % Frame bookkeeping
    ud.ultimoSlot          = '';
    ud.frameCount          = 0;

    % Inference bookkeeping
    ud.isTrackingInference = false;
    ud.inferenceStartTime  = [];

    % dt tracking
    ud.tPrev               = 0;

    % Stale frame counter
    ud.staleCount          = 0;

    % Last valid frame and outputs (avoids zeros when no new frame arrives)
    ud.lastFrame           = zeros(360, 640, 3, 'uint8');
    ud.lastS               = [0, 0];
    ud.lastBw              = 0;

    % True target-loss flag: true only when the tracker has actually given
    % up on the target (as opposed to simply waiting for a new frame or a
    % pending inference). While true, outputs are forced to a safe static
    % position instead of drifting toward a stale last-known position.
    ud.targetLost           = false;

    % Store in base workspace (survives across all S-Function callbacks)
    assignin('base','cvsfun_state', ud);
end

%% =========================================================================
function Outputs(block)

    % Load / save state via base workspace
    ud = evalin('base','cvsfun_state');

    t = block.CurrentTime;

    % Algorithm constants
    maxMisses         = 5;
    maxPredFrames     = 10;
    minConfirmCount   = 2;
    imageSizeTracking = 320;
    minConfirmScore   = 0.55;   % stricter confidence required while confirming
                                 % a brand-new target, to reject one-off false
                                 % positives before they become a tracked lock
    maxConfirmDrift   = imageSizeTracking * 0.5;  % px allowed between the
                                 % two confirmation detections (consistency)

    % Camera intrinsics
    fu    = 567;
    fv    = 566;
    alpha = 0.0587;
    Cu    = 330.7;
    Cv    = 191.3;

    % Default outputs (frameOut set later from frame or lastFrame)
    s        = [0, 0];
    bw       = 0;
    frameOut = ud.lastFrame;   % fallback: last known good frame, never black

    % Real dt
    dt = t - ud.tPrev;
    if dt <= 0,  dt = 1/30;  end
    ud.tPrev = t;

    % ── 7.1  Read next frame ─────────────────────────────────────────────
    frame = [];
    if exist(ud.slotFile, 'file')
        fid  = fopen(ud.slotFile, 'r');
        slot = fgetl(fid);
        fclose(fid);

        if ~strcmp(slot, ud.ultimoSlot)
            jpgRead = ud.jpgA;
            if strcmp(slot, 'B'),  jpgRead = ud.jpgB;  end
            try
                imgRaw = imread(jpgRead);
                if ~isempty(imgRaw)
                    frame         = imgRaw;
                    ud.ultimoSlot = slot;
                    ud.frameCount = ud.frameCount + 1;
                end
            catch
            end
        end
    end

    if isempty(frame)
        ud.staleCount = ud.staleCount + 1;
        if mod(ud.staleCount, 30) == 0
            fprintf('Sin frame nuevo desde hace %d ticks (slot=%s)\n', ud.staleCount, ud.ultimoSlot);
        end
        % Re-use last valid frame instead of outputting black
        frame = ud.lastFrame;
        if all(frame(:) == 0)
            % Still no valid frame ever received — output zeros and wait
            block.OutputPort(1).Data = double(ud.lastS);
            block.OutputPort(2).Data = double(ud.lastBw);
            block.OutputPort(3).Data = uint8(frameOut);
            assignin('base','cvsfun_state', ud);
            return;
        end
    else
        % Resize to fixed output dimensions if needed
        if size(frame,1) ~= 360 || size(frame,2) ~= 640
            frame = imresize(frame, [360, 640]);
        end
        ud.staleCount = 0;
        ud.lastFrame  = frame;
    end

    % ── 7.2/7.3  Kalman prediction ────────────────────────────────────────
    bboxes = [];

    if ud.isTracking && ~ud.isConfirming && ud.kfInitialized && ...
            ~isempty(ud.lastBboxes) && ~isempty(ud.futureInference)
        kfPred = predict(ud.kf);
        w = ud.lastBboxes(3);  h = ud.lastBboxes(4);
        bboxes = [kfPred(1)-w/2, kfPred(3)-h/2, w, h];
    end

    % While confirming a fresh detection, keep showing the last detected
    % bbox instead of nothing — avoids the rectangle blinking off during
    % the ~800ms tracking inference round-trip.
    if ud.isTracking && ud.isConfirming && ~isempty(ud.lastBboxes)
        bboxes = ud.lastBboxes;
    end

    if ~ud.isTracking && ud.kfInitialized && ~isempty(ud.lastBboxes) && ~isempty(ud.futureInference)
        if ud.predFrameCount < maxPredFrames
            kfPred            = predict(ud.kf);
            w = ud.lastBboxes(3);  h = ud.lastBboxes(4);
            bboxes            = [kfPred(1)-w/2, kfPred(3)-h/2, w, h];
            ud.lastPredBbox   = bboxes;
            ud.predFrameCount = ud.predFrameCount + 1;
        else
            bboxes = ud.lastPredBbox;
        end
    end

    % ── 7.4  Launch async YOLO inference ─────────────────────────────────
    if isempty(ud.futureInference)

        if ud.isTracking && ~isempty(ud.lastBboxes)
            % Tracking crop
            if ud.kfInitialized && ~ud.isConfirming
                state    = ud.kf.State;
                xc = state(1);  yc = state(3);
                xcLast   = ud.lastBboxes(1) + ud.lastBboxes(3)/2;
                ycLast   = ud.lastBboxes(2) + ud.lastBboxes(4)/2;
                maxDrift = imageSizeTracking * 0.5;
                xc = xcLast + max(-maxDrift, min(maxDrift, xc-xcLast));
                yc = ycLast + max(-maxDrift, min(maxDrift, yc-ycLast));
            else
                xc = ud.lastBboxes(1) + ud.lastBboxes(3)/2;
                yc = ud.lastBboxes(2) + ud.lastBboxes(4)/2;
            end

            x1 = round(xc - imageSizeTracking/2);
            y1 = round(yc - imageSizeTracking/2);
            x2 = x1 + imageSizeTracking - 1;
            y2 = y1 + imageSizeTracking - 1;
            x1 = max(1, x1);  y1 = max(1, y1);
            x2 = min(x2, size(frame,2));
            y2 = min(y2, size(frame,1));

            cropRaw = frame(y1:y2, x1:x2, :);
            cropH   = y2 - y1 + 1;
            cropW   = x2 - x1 + 1;

            if cropH ~= imageSizeTracking || cropW ~= imageSizeTracking
                crop   = imresize(cropRaw, [imageSizeTracking, imageSizeTracking]);
                scaleX = cropW / imageSizeTracking;
                scaleY = cropH / imageSizeTracking;
            else
                crop   = cropRaw;
                scaleX = 1;  scaleY = 1;
            end

            ud.sentX1  = x1;     ud.sentY1  = y1;
            ud.sentScaleX = scaleX;  ud.sentScaleY = scaleY;

            ud.futureInference     = parfeval(ud.pool, @yoloTrackingWorker, 3, ...
                                              crop, ud.modelPathTracking, ud.inputShapeTracking, 'tracking');
            ud.inferenceStartTime  = tic;
            ud.isTrackingInference = true;

        else
            % Detection: full frame
            ud.futureInference     = parfeval(ud.pool, @yoloTrackingWorker, 3, ...
                                              frame, ud.modelPathDetection, ud.inputShapeDetection, 'detection');
            ud.inferenceStartTime  = tic;
            ud.isTrackingInference = false;
        end
    end

    % ── 7.5  Collect YOLO results and update KF ───────────────────────────
    if ~isempty(ud.futureInference) && strcmp(ud.futureInference.State,'finished')

        try
            [bboxesTmp, scoresTmp, ~] = fetchOutputs(ud.futureInference);
            bboxesTmp = double(bboxesTmp);
            scoresTmp = double(scoresTmp);
        catch
            bboxesTmp = [];
            scoresTmp = [];
        end

        if ud.isTrackingInference && ~isempty(ud.inferenceStartTime)
            fprintf('Tracking inference: %.1f ms\n', toc(ud.inferenceStartTime)*1e3);
        end
        ud.inferenceStartTime  = [];
        ud.isTrackingInference = false;
        ud.futureInference     = [];

        % ── Tracking branch ───────────────────────────────────────────────
        if ud.isTracking

            % ── Sanity check: reject detections that jump implausibly far
            % from the last known position. The tracking crop is only
            % imageSizeTracking px wide, so a real target can't have moved
            % more than a fraction of that between consecutive ticks —
            % anything else is very likely a false positive picked up
            % inside the crop (another object, shadow, noise, etc).
            %
            % IMPORTANT: the allowed jump grows with consecutive misses,
            % and a candidate that reappears in roughly the same spot two
            % ticks in a row is accepted even if far away — a random false
            % positive rarely repeats in the same location, but a real
            % target that moved while we were missing it will.
            if ~isempty(bboxesTmp)
                bboxesRaw      = bboxesTmp;
                bboxesRaw(:,1) = bboxesTmp(:,1)*ud.sentScaleX + ud.sentX1 - 1;
                bboxesRaw(:,2) = bboxesTmp(:,2)*ud.sentScaleY + ud.sentY1 - 1;
                bboxesRaw(:,3) = bboxesTmp(:,3)*ud.sentScaleX;
                bboxesRaw(:,4) = bboxesTmp(:,4)*ud.sentScaleY;

                xcRaw = bboxesRaw(1,1) + bboxesRaw(1,3)/2;
                ycRaw = bboxesRaw(1,2) + bboxesRaw(1,4)/2;

                if ~isempty(ud.lastBboxes)
                    xcLastTr  = ud.lastBboxes(1) + ud.lastBboxes(3)/2;
                    ycLastTr  = ud.lastBboxes(2) + ud.lastBboxes(4)/2;
                    distTrack = norm([xcRaw-xcLastTr, ycRaw-ycLastTr]);

                    wRatio = bboxesRaw(1,3) / max(ud.lastBboxes(3), 1e-3);
                    hRatio = bboxesRaw(1,4) / max(ud.lastBboxes(4), 1e-3);
                    sizeJumpOk = wRatio > 0.5 && wRatio < 2 && ...
                                 hRatio > 0.5 && hRatio < 2;

                    % Threshold relaxes the longer we've gone without a
                    % confirmed detection (target may have legitimately
                    % drifted while we were missing it).
                    missGrowth   = 1 + 0.5 * ud.missCount;   % +50% per consecutive miss
                    maxTrackJump = imageSizeTracking * 0.2 * missGrowth;

                    isFarJump = distTrack > maxTrackJump;

                    % Consistency check: does this candidate match the one
                    % we rejected on the PREVIOUS tick (within a small
                    % tolerance)? If so, two independent inferences agree
                    % on the same new location — treat it as real.
                    reappeared = false;
                    if isFarJump && ~isempty(ud.lastRejectedCenter)
                        reappearDist = norm([xcRaw, ycRaw] - ud.lastRejectedCenter);
                        if reappearDist < imageSizeTracking * 0.15
                            reappeared = true;
                        end
                    end

                    if isFarJump && ~sizeJumpOk
                        % Size is implausible regardless — always reject.
                        fprintf('Tracking: false positive rejected (dist=%.0f, wRatio=%.2f, hRatio=%.2f)\n', ...
                            distTrack, wRatio, hRatio);
                        ud.lastRejectedCenter = [xcRaw, ycRaw];
                        bboxesTmp = [];
                    elseif isFarJump && ~reappeared
                        fprintf('Tracking: distant candidate rejected (dist=%.0f > %.0f, miss#%d) — waiting for confirmation\n', ...
                            distTrack, maxTrackJump, ud.missCount);
                        ud.lastRejectedCenter = [xcRaw, ycRaw];
                        bboxesTmp = [];
                    else
                        if reappeared
                            disp('Tracking: distant candidate confirmed on 2nd appearance — accepting as real target')
                        end
                        ud.lastRejectedCenter = [];
                    end
                else
                    ud.lastRejectedCenter = [];
                end
            end

            if ~isempty(bboxesTmp)
                bboxes      = bboxesTmp;
                bboxes(:,1) = bboxesTmp(:,1)*ud.sentScaleX + ud.sentX1 - 1;
                bboxes(:,2) = bboxesTmp(:,2)*ud.sentScaleY + ud.sentY1 - 1;
                bboxes(:,3) = bboxesTmp(:,3)*ud.sentScaleX;
                bboxes(:,4) = bboxesTmp(:,4)*ud.sentScaleY;
                ud.lastBboxes = bboxes(1,:);

                xc = ud.lastBboxes(1) + ud.lastBboxes(3)/2;
                yc = ud.lastBboxes(2) + ud.lastBboxes(4)/2;

                if ud.isConfirming
                    % Require a reasonably confident detection AND that it
                    % lands close to the position seen on the previous
                    % confirmation attempt. A genuine target stays roughly
                    % put across 2 consecutive frames; a one-off false
                    % positive (shadow, noise) usually does not reappear in
                    % the same exact spot with similarly high confidence.
                    scoreOk = isempty(scoresTmp) || scoresTmp(1) >= minConfirmScore;

                    consistentOk = true;
                    if ~isempty(ud.confirmCenter)
                        confirmDrift = norm([xc, yc] - ud.confirmCenter);
                        consistentOk = confirmDrift <= maxConfirmDrift;
                    end

                    if scoreOk && consistentOk
                        ud.confirmCenter = [xc, yc];
                        ud.confirmCount  = ud.confirmCount + 1;
                        if ud.confirmCount >= minConfirmCount
                            disp('Target confirmed. Starting tracking...')
                            ud.isConfirming  = false;
                            ud.confirmCenter = [];
                            ud.kf = trackingKF('MotionModel','2D Constant Velocity', ...
                                               'State',[xc;0;yc;0], ...
                                               'MeasurementNoise',10, ...
                                               'ProcessNoise',ud.adaptiveQ);
                            ud.kfInitialized = true;
                            ud.lastCenter    = [xc, yc];
                        end
                    else
                        % Low confidence or inconsistent with the previous
                        % confirmation hit — this was likely a fluke.
                        % Abort the confirmation attempt entirely and go
                        % back to full-frame detection rather than locking
                        % onto a false target.
                        fprintf(['Confirmation rejected (score=%.2f scoreOk=%d, ', ...
                                 'drift consistentOk=%d) — back to detection\n'], ...
                                 scoresTmp(1)*~isempty(scoresTmp), scoreOk, consistentOk);
                        ud.isTracking    = false;
                        ud.isConfirming  = false;
                        ud.confirmCount  = 0;
                        ud.confirmCenter = [];
                        ud.lastBboxes    = [];
                        bboxes           = [];
                        ud.targetLost    = true;
                    end
                else
                    if ~ud.kfInitialized
                        ud.kf = trackingKF('MotionModel','2D Constant Velocity', ...
                                           'State',[xc;0;yc;0], ...
                                           'MeasurementNoise',10, ...
                                           'ProcessNoise',ud.adaptiveQ);
                        ud.kfInitialized = true;
                        ud.lastCenter    = [xc, yc];
                    else
                        if ~isempty(ud.lastCenter)
                            accel = norm([xc,yc] - ud.lastCenter) / dt;
                            if     accel > 70,  ud.adaptiveQ = 300;  ud.kf.MeasurementNoise = 5;
                            elseif accel > 15,  ud.adaptiveQ = 200;  ud.kf.MeasurementNoise = 10;
                            else,               ud.adaptiveQ = 50;   ud.kf.MeasurementNoise = 20;
                            end
                            ud.kf.ProcessNoise = ud.adaptiveQ;
                        end
                        ud.lastCenter = [xc, yc];
                        correct(ud.kf, [xc, yc]);
                    end
                end

                ud.missCount      = 0;
                ud.predFrameCount = 0;
                ud.lastPredBbox   = [];

                if ~ud.isConfirming
                    state  = ud.kf.State;
                    bboxes = [state(1)-ud.lastBboxes(3)/2, state(3)-ud.lastBboxes(4)/2, ...
                              ud.lastBboxes(3), ud.lastBboxes(4)];
                end

            else
                ud.missCount = ud.missCount + 1;
                if ud.isConfirming
                    disp('Confirmation failed. Back to detection...')
                    ud.isTracking   = false;  ud.isConfirming = false;
                    ud.confirmCount = 0;       ud.missCount    = 0;
                    ud.lastBboxes   = [];      bboxes          = [];
                    ud.lastCenter   = [];      ud.lastRejectedCenter = [];
                    ud.confirmCenter = [];
                    ud.targetLost    = true;
                else
                    fprintf('Target not detected. Attempt %d/%d\n', ud.missCount, maxMisses);
                    if ud.missCount >= maxMisses
                        disp('Target lost. Switching to detection...')
                        ud.isTracking     = false;
                        ud.missCount      = 0;  ud.predFrameCount = 0;
                        ud.confirmCount   = 0;  ud.lastCenter     = [];
                        ud.adaptiveQ      = 200;
                        bboxes            = [];
                        ud.lastRejectedCenter = [];
                        ud.targetLost     = true;
                    else
                        % Missed detection but not yet lost: keep showing
                        % the KF prediction / last known bbox instead of
                        % letting the rectangle disappear (avoids flicker).
                        if ud.kfInitialized
                            state  = ud.kf.State;
                            bboxes = [state(1)-ud.lastBboxes(3)/2, state(3)-ud.lastBboxes(4)/2, ...
                                      ud.lastBboxes(3), ud.lastBboxes(4)];
                        else
                            bboxes = ud.lastBboxes;
                        end
                    end
                end
            end

        % ── Detection branch ──────────────────────────────────────────────
        else
            if ~isempty(bboxesTmp)
                xcDet = bboxesTmp(1,1) + bboxesTmp(1,3)/2;
                ycDet = bboxesTmp(1,2) + bboxesTmp(1,4)/2;

                isValid = true;
                if ~isempty(ud.lastBboxes)
                    xcLast = ud.lastBboxes(1) + ud.lastBboxes(3)/2;
                    ycLast = ud.lastBboxes(2) + ud.lastBboxes(4)/2;
                    dist   = norm([xcDet-xcLast, ycDet-ycLast]);
                    if dist > imageSizeTracking * 2.0
                        disp('Warning: false positive rejected')
                        isValid         = false;
                        ud.confirmCount = 0;
                        bboxes          = ud.lastPredBbox;
                    end
                end

                if isValid
                    disp('Target detected. Waiting for confirmation...')
                    ud.lastBboxes    = double(bboxesTmp(1,:));
                    bboxes           = double(bboxesTmp);
                    ud.isTracking    = true;
                    ud.isConfirming  = true;
                    ud.confirmCount  = 1;
                    ud.confirmCenter = [xcDet, ycDet];
                    ud.missCount     = 0;  ud.predFrameCount = 0;
                    ud.lastPredBbox  = [];
                    ud.targetLost    = false;
                end
            else
                ud.confirmCount = 0;
                if ud.kfInitialized
                    ud.missCount = ud.missCount + 1;
                    if ud.missCount >= maxMisses * 3
                        disp('No target found. Resetting tracker...')
                        ud.kfInitialized  = false;
                        ud.lastBboxes     = [];  ud.lastPredBbox = [];
                        bboxes            = [];  ud.missCount    = 0;
                        ud.predFrameCount = 0;   ud.lastCenter   = [];
                        ud.adaptiveQ      = 200;
                        ud.targetLost     = true;
                    end
                end
            end
        end
    end

    % ── Safety net: guarantee a continuous bbox while tracking ─────────────
    % Whatever branch ran above, if we are still in tracking mode and have
    % *some* last known bbox/KF state, never let `bboxes` end up empty.
    % This is what the IBVS control loop needs: a bbox on every single tick,
    % not just on ticks where a fresh detection/tracking inference landed.
    isPrediction = false;
    if ud.isTracking && (isempty(bboxes) || size(bboxes,2) < 4)
        if ud.kfInitialized && ~isempty(ud.lastBboxes)
            state  = ud.kf.State;
            bboxes = [state(1)-ud.lastBboxes(3)/2, state(3)-ud.lastBboxes(4)/2, ...
                      ud.lastBboxes(3), ud.lastBboxes(4)];
            isPrediction = true;
        elseif ~isempty(ud.lastBboxes)
            bboxes = ud.lastBboxes;
            isPrediction = true;
        end
    end

    % ── Extract and normalise outputs ─────────────────────────────────────
    if ~isempty(bboxes) && size(bboxes,2) >= 4
        u  = bboxes(1,1) + bboxes(1,3)/2;
        v  = bboxes(1,2) + bboxes(1,4)/2;
        bw = bboxes(1,3);

        s = [(u-Cu)/(fu*alpha),  (v-Cv)/fv];

        if ud.isConfirming
            color = 'yellow';
        elseif ~ud.isTracking && ud.predFrameCount >= maxPredFrames
            color = 'green';
        elseif isPrediction
            % Showing KF/last-known prediction, not a fresh detection this
            % tick — still tracking, just visually distinct.
            color = 'cyan';
        else
            color = 'red';
        end
        frameOut = insertShape(frame, 'Rectangle', bboxes(1,:), 'Color', color, 'LineWidth', 2);
    else
        frameOut = frame;
    end
    
    % Si el objetivo se perdió de verdad (tracker lo dio por perdido),
    % forzar una posición estática segura en vez de seguir enviando la
    % última posición conocida indefinidamente.
    if ud.targetLost
        s         = [0, 0];
        bw        = 100;
        ud.lastS  = s;
        ud.lastBw = bw;
    elseif any(s ~= 0) || bw ~= 0
        % Detección/tracking válido en este tick: actualizar el último
        % valor conocido.
        ud.lastS  = s;
        ud.lastBw = bw;
    else
        % Aún no ha llegado un frame nuevo o la inferencia sigue corriendo:
        % mantener el último valor válido conocido (sin marcar pérdida).
        s  = ud.lastS;
        bw = ud.lastBw;
    end

    block.OutputPort(1).Data = double(s);
    block.OutputPort(2).Data = double(bw);
    block.OutputPort(3).Data = uint8(frameOut);

    % Persist updated state
    assignin('base','cvsfun_state', ud);
end

%% =========================================================================
function Terminate(block) %#ok<INUSD>
% Clean up the state variable from the base workspace
    try
        evalin('base','clear cvsfun_state');
    catch
    end
end