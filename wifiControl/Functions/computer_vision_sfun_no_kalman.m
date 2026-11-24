function computer_vision_sfun_no_kalman(block)
% COMPUTER_VISION_SFUN_NO_KALMAN  Level-2 MATLAB S-Function for drone tracking.
% Sin Kalman: s solo sale de detecciones YOLO reales, nunca de una prediccion.
%
% Outputs:
%   1 - s        : normalised [x, y] image error for IBVS      [1x2 double]
%   2 - bw       : bbox width in pixels                        [scalar double]
setup(block);
end

%% =========================================================================
function setup(block)

    block.NumInputPorts  = 0;
    block.NumOutputPorts = 2;

    % Output 1: s [1x2]
    block.OutputPort(1).Dimensions = [1 2];
    block.OutputPort(1).DatatypeID = 0;
    block.OutputPort(1).Complexity = 'Real';

    % Output 2: bw (scalar)
    block.OutputPort(2).Dimensions = 1;
    block.OutputPort(2).DatatypeID = 0;
    block.OutputPort(2).Complexity = 'Real';

    block.SampleTimes = [0.03 0];

    block.RegBlockMethod('Start',    @Start);
    block.RegBlockMethod('Outputs',  @Outputs);
    block.RegBlockMethod('Terminate',@Terminate);
end

%% =========================================================================
function Start(block) %#ok<INUSD>

    ud = struct();

    ud.pool                = evalin('base','pool');
    ud.modelPathDetection  = evalin('base','modelPathDetection');
    ud.modelPathTracking   = evalin('base','modelPathTracking');
    ud.inputShapeDetection = evalin('base','inputShapeDetection');
    ud.inputShapeTracking  = evalin('base','inputShapeTracking');
    ud.jpgA                = evalin('base','jpgA');
    ud.jpgB                = evalin('base','jpgB');
    ud.slotFile            = evalin('base','slotFile');

    % Tracker state (sin Kalman)
    ud.isTracking          = false;
    ud.isConfirming        = false;
    ud.missCount           = 0;
    ud.confirmCount        = 0;
    ud.lastBboxes          = [];
    ud.lastRejectedCenter  = [];
    ud.confirmCenter       = [];
    ud.futureInference     = [];
    ud.targetLost          = false;

    % Crop-to-frame mapping
    ud.sentX1              = 1;
    ud.sentY1              = 1;
    ud.sentScaleX          = 1;
    ud.sentScaleY          = 1;

    % Frame bookkeeping
    ud.ultimoSlot          = '';
    ud.frameCount          = 0;
    ud.staleCount          = 0;

    % Inference bookkeeping
    ud.isTrackingInference = false;
    ud.inferenceStartTime  = [];

    % Last valid s/bw (Zero-Order Hold entre detecciones)
    ud.lastS               = [0, 0];
    ud.lastBw              = 0;

    assignin('base','cvsfun_state', ud);
end

%% =========================================================================
function Outputs(block)

    ud = evalin('base','cvsfun_state');

    % Algorithm constants
    maxMisses             = 1;
    imageSizeTracking     = 320;
    minConfirmCount       = 2;
    minConfirmScore       = 0.3;
    maxConfirmDrift       = imageSizeTracking * 0.5;
    maxStaleWhileTracking = 10;   % ticks ~450ms sin frame -> reset
    maxTrackJump          = imageSizeTracking * 0.4; % salto max aceptado sin prediccion

    % Camera intrinsics
    Dron = 1;
    if Dron==3
        fu    = 566.35; %562.88
        fv    = 564.89; % 560.74
        Cu    = 330.2733; %332.33
        Cv    = 184.9;    % 163.94
    elseif Dron==1
        fu    = 562.88;
        fv    = 560.74;
        %alpha = 0.0878;
        Cu    = 332.33;
        Cv    = 184.9;   
    end
    alpha = fv/fu;


    % Default outputs
    s  = [0, 0];
    bw = 0;

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

        if ud.isTracking && ud.staleCount >= maxStaleWhileTracking
            disp('[SFUN] No new frame while tracking — resetting tracker.')
            ud.isTracking         = false;
            ud.isConfirming       = false;
            ud.missCount          = 0;
            ud.confirmCount       = 0;
            ud.lastBboxes         = [];
            ud.lastRejectedCenter = [];
            ud.confirmCenter      = [];
            ud.targetLost         = true;
            ud.lastS              = [0, 0];
            ud.lastBw             = 0;
            if ~isempty(ud.futureInference)
                cancel(ud.futureInference);
                ud.futureInference = [];
            end
        end

        block.OutputPort(1).Data = double(ud.lastS);
        block.OutputPort(2).Data = double(ud.lastBw);
        assignin('base','cvsfun_state', ud);
        return;
    else
        ud.staleCount = 0;
        ud.targetLost = false;
    end

    bboxes = [];

    % ── 7.2  Launch async YOLO inference ─────────────────────────────────
    if isempty(ud.futureInference)

        if ud.isTracking && ~isempty(ud.lastBboxes)
            % Tracking crop: centrado en la ULTIMA deteccion real, sin prediccion
            xc = ud.lastBboxes(1) + ud.lastBboxes(3)/2;
            yc = ud.lastBboxes(2) + ud.lastBboxes(4)/2;

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

            ud.sentX1     = x1;     ud.sentY1     = y1;
            ud.sentScaleX = scaleX; ud.sentScaleY = scaleY;

            ud.futureInference     = parfeval(ud.pool, @yoloTrackingWorker, 3, ...
                                              crop, ud.modelPathTracking, ud.inputShapeTracking, 'tracking');
            ud.inferenceStartTime  = tic;
            ud.isTrackingInference = true;

        else
            % Deteccion: frame completo
            ud.futureInference     = parfeval(ud.pool, @yoloTrackingWorker, 3, ...
                                              frame, ud.modelPathDetection, ud.inputShapeDetection, 'detection');
            ud.inferenceStartTime  = tic;
            ud.isTrackingInference = false;
        end
    end

    % ── 7.3  Collect YOLO results ─────────────────────────────────────────
    freshDetection = false;   % true SOLO si llega una deteccion real este tick

    if ~isempty(ud.futureInference) && strcmp(ud.futureInference.State,'finished')

        try
            [bboxesTmp, scoresTmp, ~] = fetchOutputs(ud.futureInference);
            bboxesTmp = double(bboxesTmp);
            scoresTmp = double(scoresTmp);
        catch ME
            bboxesTmp = [];
            scoresTmp = [];
            fprintf('[WORKER ERROR] %s\n', ME.message);
        end

        if ud.isTrackingInference && ~isempty(ud.inferenceStartTime)
            fprintf('Tracking inference: %.1f ms\n', toc(ud.inferenceStartTime)*1e3);
        end
        ud.inferenceStartTime  = [];
        ud.isTrackingInference = false;
        ud.futureInference     = [];

        % ── Tracking branch ───────────────────────────────────────────────
        if ud.isTracking

            if ~isempty(bboxesTmp)
                bboxesRaw      = bboxesTmp;
                bboxesRaw(:,1) = bboxesTmp(:,1)*ud.sentScaleX + ud.sentX1 - 1;
                bboxesRaw(:,2) = bboxesTmp(:,2)*ud.sentScaleY + ud.sentY1 - 1;
                bboxesRaw(:,3) = bboxesTmp(:,3)*ud.sentScaleX;
                bboxesRaw(:,4) = bboxesTmp(:,4)*ud.sentScaleY;

                xcRaw = bboxesRaw(1,1) + bboxesRaw(1,3)/2;
                ycRaw = bboxesRaw(1,2) + bboxesRaw(1,4)/2;

                % Rechazo de saltos implausibles (sin gracia de Kalman, umbral fijo)
                if ~isempty(ud.lastBboxes)
                    xcLast    = ud.lastBboxes(1) + ud.lastBboxes(3)/2;
                    ycLast    = ud.lastBboxes(2) + ud.lastBboxes(4)/2;
                    distTrack = norm([xcRaw-xcLast, ycRaw-ycLast]);
                    wRatio    = bboxesRaw(1,3) / max(ud.lastBboxes(3), 1e-3);
                    hRatio    = bboxesRaw(1,4) / max(ud.lastBboxes(4), 1e-3);
                    sizeJumpOk = wRatio > 0.4 && wRatio < 2.5 && hRatio > 0.4 && hRatio < 2.5;
                    isFarJump  = distTrack > maxTrackJump;

                    reappeared = false;
                    if isFarJump && ~isempty(ud.lastRejectedCenter)
                        if norm([xcRaw,ycRaw] - ud.lastRejectedCenter) < imageSizeTracking*0.15
                            reappeared = true;
                        end
                    end

                    if isFarJump && ~sizeJumpOk
                        fprintf('Tracking: false positive rejected (dist=%.0f wR=%.2f hR=%.2f)\n', distTrack, wRatio, hRatio);
                        ud.lastRejectedCenter = [xcRaw, ycRaw];
                        bboxesTmp = [];
                    elseif isFarJump && ~reappeared
                        fprintf('Tracking: distant candidate rejected (dist=%.0f > %.0f)\n', distTrack, maxTrackJump);
                        ud.lastRejectedCenter = [xcRaw, ycRaw];
                        bboxesTmp = [];
                    else
                        if reappeared
                            disp('Tracking: distant candidate confirmed on 2nd appearance')
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
                    scoreOk      = isempty(scoresTmp) || scoresTmp(1) >= minConfirmScore;
                    consistentOk = true;
                    if ~isempty(ud.confirmCenter)
                        consistentOk = norm([xc,yc] - ud.confirmCenter) <= maxConfirmDrift;
                    end

                    if scoreOk && consistentOk
                        ud.confirmCenter = [xc, yc];
                        ud.confirmCount  = ud.confirmCount + 1;
                        if ud.confirmCount >= minConfirmCount
                            disp('Target confirmed. Starting tracking...')
                            ud.isConfirming  = false;
                            ud.confirmCenter = [];
                            freshDetection   = true;
                        end
                    else
                        fprintf('Confirmation rejected (scoreOk=%d consistentOk=%d)\n', scoreOk, consistentOk);
                        ud.isTracking    = false;  ud.isConfirming  = false;
                        ud.confirmCount  = 0;       ud.confirmCenter = [];
                        ud.lastBboxes    = [];       bboxes          = [];
                        ud.targetLost    = true;
                    end
                else
                    % Deteccion real confirmada este tick: no hay prediccion previa
                    freshDetection = true;
                end

                ud.missCount = 0;

            else
                ud.missCount = ud.missCount + 1;
                if ud.isConfirming
                    disp('Confirmation failed. Back to detection...')
                    ud.isTracking    = false;  ud.isConfirming       = false;
                    ud.confirmCount  = 0;       ud.missCount          = 0;
                    ud.lastBboxes    = [];       ud.lastRejectedCenter = [];
                    ud.confirmCenter = [];       bboxes                = [];
                    ud.targetLost    = true;
                else
                    fprintf('Target not detected. Attempt %d/%d\n', ud.missCount, maxMisses);
                    if ud.missCount >= maxMisses
                        disp('Target lost. Switching to detection...')
                        ud.isTracking         = false;
                        ud.missCount          = 0;
                        ud.confirmCount       = 0;
                        ud.lastRejectedCenter = [];
                        bboxes                = [];
                        ud.targetLost         = true;
                    end
                end
            end

        % ── Detection branch ──────────────────────────────────────────────
        else
            if ~isempty(bboxesTmp)
                xcDet = bboxesTmp(1,1) + bboxesTmp(1,3)/2;
                ycDet = bboxesTmp(1,2) + bboxesTmp(1,4)/2;

                disp('Target detected. Waiting for confirmation...')
                ud.lastBboxes    = double(bboxesTmp(1,:));
                bboxes           = double(bboxesTmp);
                ud.isTracking    = true;
                ud.isConfirming  = true;
                ud.confirmCount  = 1;
                ud.confirmCenter = [xcDet, ycDet];
                ud.missCount     = 0;
                ud.targetLost    = false;
            end
        end
    end

    % ── Compute s/bw from bbox ────────────────────────────────────────────
    if freshDetection && ~isempty(bboxes) && size(bboxes,2) >= 4
        u  = bboxes(1,1) + bboxes(1,3)/2;
        v  = bboxes(1,2) + bboxes(1,4)/2;
        bw = bboxes(1,3);
        s  = [(u-Cu)/(fu*alpha), (v-Cv)/fv];
    end

    % ── isValid ──────────────────────────────────────────────────────────
    % En cualquier otro caso (esperando inferencia, target perdido,
    % confirmando)se mantiene el ultimo s conocido (ZOH),
    if ud.targetLost
        s       = [0, 0];
        bw      = 0;
        ud.lastS  = s;
        ud.lastBw = bw;
    else 
        if freshDetection
            ud.lastS  = s;
            ud.lastBw = bw;
        else
            s       = ud.lastS;
            bw      = ud.lastBw;
        end
    end

    block.OutputPort(1).Data = double(s);
    block.OutputPort(2).Data = double(bw);

    assignin('base','cvsfun_state', ud);
end

%% =========================================================================
function Terminate(block) %#ok<INUSD>
    try
        evalin('base','clear cvsfun_state');
    catch
    end
end
