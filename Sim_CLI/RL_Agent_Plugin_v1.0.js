// DMMS.R에서 RL 에이전트와 통신하기 위한 JavaScript 플러그인
// Version: 1.0.0 (Specification for G2P2C Agent Applied)
// 이 스크립트는 매분 호출되는 `runIteration` 함수 내에서 5분 간격으로 상태를
// FastAPI 서버에 전송하고, 받은 인슐린 주입률을 다음 5분 동안 적용한다.
// Changes:
// - Implemented 5-minute interval logic for API communication and history.
// - Added 'hour' and 'meal' data to agent state.
// - CGM history stores the last value of each 5-minute interval.
// - Insulin history stores the U/h rate applied during the previous 5-minute interval.
// - Insulin action from agent is applied consistently over the 5-minute interval.

// --- 설정 변수 ---
var AGENT_COMMUNICATION_METHOD = 'WEB_SERVICE'; // 웹 서비스 방식 사용
var AGENT_API_URL = "http://127.0.0.1:5000"; // FastAPI 서버 주소
var WEB_REQUEST_TIMEOUT_MS = 5000; // 웹 서비스 타임아웃 (밀리초)

var INSULIN_NAME_DEFINED_IN_DMMS = "PPO_Agent_Insulin"; // DMMS.R 'Insulin Definitions'에서 정의한 이름 
var INSULIN_TARGET_KEY = "sqCustomInsulin1"; // ** 사용자가 정의한 순서에 따라 sqCustomInsulin2 등이 될 수 있음. 
var FALLBACK_INSULIN_KEY = "sqInsulinNormalBolus"; // sqCustomInsulinX 키 실패 시 사용할 표준 볼루스 키

var PMOL_PER_UNIT = 6000.0;
var FEATURE_HISTORY_LENGTH = 12; // Represents 12 * 5-minute intervals = 60 minutes
var ACTION_MIN_U_PER_H = 0.0;
var ACTION_MAX_U_PER_H = 5.0; 
var USE_FEAT_VECTOR = false; // 추가 특징 벡터 사용 여부 - G2P2C는 정규화된 시간을 사용 (hour로 전달)

// --- 전역 변수 (상태 관리용) ---
var debugLog = "";

var bgHistory = []; // 5-minute interval CGM data
var insulinActionHistory = []; // 5-minute interval applied insulin (U/h)

var isInitialized = false;
var lastAppliedAction_UperH = 0.0; // Action applied in the *current* 5-min interval (U/h)
                                   // This is also used for insulinActionHistory for the *next* cycle.
var current_rate_for_minute_application_U_per_H = 0.0; // Actual U/h rate to apply each minute

// --- DMMS.R가 호출하는 필수 함수들 ---

// 시뮬레이터가 요구하는 입력 시그널 개수 반환 (CGM 한 개)
function numSignals() { return 1; }

// 각 시그널에 대한 설명 문자열 반환
function signalDescription(signalIndex) {
    if (signalIndex === 0) { return "Blood Glucose Input (CGM)"; }
    return "Unknown Signal";
}

// 시뮬레이션 시작 시 호출되어 내부 상태를 초기화한다.
function initialize(popName, subjName, simDuration, configDir, baseResultsDir, simName) {
    debugLog = "[Initialize] Attempting initialization for subject: " + subjName + ", Sim: " + simName + " (WebService Mode, 5-min interval logic).";
    try {
        bgHistory = []; // Clear history for new simulation
        insulinActionHistory = [];

        // Initialize history arrays with 0.0 to FEATURE_HISTORY_LENGTH
        // This ensures they are pre-filled for the first API call if needed.
        // Actual meaningful values will be pushed at 5-min intervals.
        for (var i = 0; i < FEATURE_HISTORY_LENGTH; i++) {
            bgHistory.push(0.0);
            insulinActionHistory.push(0.0);
        }

        lastAppliedAction_UperH = 0.0;
        current_rate_for_minute_application_U_per_H = 0.0;
        isInitialized = true;
        debugLog += "\n[Initialize] Initialization complete. isInitialized set to true. History arrays pre-filled.";
    } catch (e) {
        debugLog += "\n[Initialize] !!! CRITICAL ERROR IN INITIALIZE: " + e;
        isInitialized = false;
    }
}

// 매 분 실행되며 5분 간격으로 에이전트와 통신해 인슐린 값을 결정한다.
function runIteration(subjObject, sensorSigArray, nextMealObject, nextExerciseObject, timeObject, modelInputsToModObject) {
    if (!isInitialized) {
        if (debugLog.indexOf("runIteration called before initialize!") === -1) {
            debugLog = "[Error] runIteration called before initialize!";
        }
        return;
    }

    var currentMinuteAbs = NaN; // Absolute minutes from simulation start
    var currentCGM = NaN;
    var currentHour = NaN; // Hour of the day (0-23)
    var timeToNextMealSteps = 0.0; // Time until next meal in 5-min steps (0 if unknown)

    // --- 1. 시간 및 현재 CGM 값 가져오기 (매분 실행) ---
    if (timeObject && typeof timeObject === 'object' && timeObject.minutesPastSimStart !== undefined && isFinite(timeObject.minutesPastSimStart)) {
        currentMinuteAbs = timeObject.minutesPastSimStart;
    } else {
        debugLog += "\n[RunIteration] Invalid timeObject or minutesPastSimStart. Skipping iteration.";
        return;
    }
    debugLog = "[RunIteration] Minute: " + currentMinuteAbs;

    currentCGM = sensorSigArray[0];
    if (isNaN(currentCGM) || !isFinite(currentCGM)) {
        debugLog += "\n[Warning] Invalid CGM value received: " + currentCGM + ". Using 0 for this reading.";
        currentCGM = 0.0;
    }
    debugLog += ", CGM: " + currentCGM.toFixed(2);

    // --- 2. 인슐린 적용 (매분 실행) ---
    // 에이전트가 5분마다 결정한 주입률(current_rate_for_minute_application_U_per_H)을 매분 적용
    var dose_pmol_per_Min_current = current_rate_for_minute_application_U_per_H / 60.0 * PMOL_PER_UNIT;

    var keyToUseForInsulin = "";
    if (INSULIN_TARGET_KEY in modelInputsToModObject) {
        keyToUseForInsulin = INSULIN_TARGET_KEY;
    } else if (FALLBACK_INSULIN_KEY in modelInputsToModObject) {
        keyToUseForInsulin = FALLBACK_INSULIN_KEY;
        debugLog += "\n[Warning] Target key '" + INSULIN_TARGET_KEY + "' not found! Using fallback key '" + keyToUseForInsulin + "'.";
    } else {
        debugLog += "\n[Error] Neither target key '" + INSULIN_TARGET_KEY + "' nor fallback key '" + FALLBACK_INSULIN_KEY + "' found! Cannot apply insulin.";
    }

    if (keyToUseForInsulin !== "") {
        modelInputsToModObject[keyToUseForInsulin] = dose_pmol_per_Min_current;
        // debugLog += "\nApplied " + current_rate_for_minute_application_U_per_H.toFixed(4) + " U/h (" + dose_pmol_per_Min_current.toFixed(4) + " pmol/min) to " + keyToUseForInsulin;
    }


    // --- 3. 5분 간격 처리 로직 (API 호출, 이력 업데이트 등) ---
    if (currentMinuteAbs % 5 === 0) {
        debugLog += "\n--- 5-Minute Interval Boundary (Minute: " + currentMinuteAbs + ") ---";

        // 3a. 현재 시간(hour) 및 식사량(meal) 정보 추출
        currentHour = Math.floor(timeObject.minutesPastMidnight / 60) % 24;
        debugLog += "\n Current Hour: " + currentHour;

        // 다음 식사까지 남은 시간 정보를 계산 (분 단위 → 5분 스텝)
        var minutesUntilNextMeal = 0;
        if (nextMealObject && typeof nextMealObject === 'object') {
            if (nextMealObject.minutesUntilNextMeal !== undefined &&
                isFinite(nextMealObject.minutesUntilNextMeal)) {
                minutesUntilNextMeal = nextMealObject.minutesUntilNextMeal;
            } else if (nextMealObject.minutes_to_next_meal !== undefined &&
                       isFinite(nextMealObject.minutes_to_next_meal)) {
                minutesUntilNextMeal = nextMealObject.minutes_to_next_meal;
            }
        }
        timeToNextMealSteps = minutesUntilNextMeal / 5.0; // 남은 시간을 5분 스텝으로 전달
        if (minutesUntilNextMeal > 0) {
            debugLog += "\n Time until next meal: " + minutesUntilNextMeal.toFixed(1) + " min";
            debugLog += "\n No upcoming meal info or meal announcements disabled.";
        }


        // 3b. 이력 배열 업데이트 (5분 간격 데이터)
        // CGM 이력: 현재 CGM 값 (5분 간격의 마지막 값)을 사용
        bgHistory.push(currentCGM);
        if (bgHistory.length > FEATURE_HISTORY_LENGTH) {
            bgHistory.shift();
        }
        debugLog += "\n Updated bgHistory (last 5min CGM): " + currentCGM.toFixed(2) + ", Length: " + bgHistory.length;

        // 인슐린 이력: *이전* 5분 동안 적용되었던 인슐린 액션 (lastAppliedAction_UperH)
        insulinActionHistory.push(lastAppliedAction_UperH);
        if (insulinActionHistory.length > FEATURE_HISTORY_LENGTH) {
            insulinActionHistory.shift();
        }
        debugLog += "\n Updated insulinActionHistory (previous 5min U/h): " + lastAppliedAction_UperH.toFixed(4) + ", Length: " + insulinActionHistory.length;

        // 3c. 에이전트 상태 준비
        // slice()로 복사본을 만들어 전달 (prepareAgentState에서 변경하지 않도록)
        var agentState = prepareAgentState(
            bgHistory.slice(),
            insulinActionHistory.slice(),
            currentHour,
            timeToNextMealSteps,
            timeObject,   // 추가 정보 전달용 (필요시 prepareAgentState에서 활용)
            nextMealObject, // 추가 정보 전달용
            nextExerciseObject // 추가 정보 전달용
        );
        debugLog += "\n Prepared Agent State for API: " + JSON.stringify(agentState);

        // 3d. 에이전트와 통신 (웹 서비스 - HTTP POST)
        var agentDecidedAction_U_per_Hour = 0.0; // 기본값
        var communicationSuccess = false;

        try {
            var requestBody = JSON.stringify(agentState);
            var contentType = "application/json";
            var success = httpWebServiceInvoker.performPostRequest(AGENT_API_URL + "/env_step", requestBody, contentType, WEB_REQUEST_TIMEOUT_MS);

            if (success && !httpWebServiceInvoker.timeout() && httpWebServiceInvoker.responseStatusCode() >= 200 && httpWebServiceInvoker.responseStatusCode() < 300) {
                var responseBody = httpWebServiceInvoker.responseBody();
                debugLog += "\n Raw API Response: " + responseBody;
                try {
                    var parsedResponse = JSON.parse(responseBody);
                    if (parsedResponse && typeof parsedResponse.insulin_action_U_per_h === 'number' && isFinite(parsedResponse.insulin_action_U_per_h)) {
                        agentDecidedAction_U_per_Hour = parsedResponse.insulin_action_U_per_h;
                        communicationSuccess = true;
                        debugLog += "\n Parsed Action from Agent (U/h): " + agentDecidedAction_U_per_Hour.toFixed(4);
                    } else {
                        debugLog += "\n [Error] Invalid JSON structure/value in API response: " + responseBody;
                    }
                } catch (parseError) {
                    debugLog += "\n [Error] Failed to parse JSON API response: " + parseError + "\nResponse body: " + responseBody;
                }
            } else {
                debugLog += "\n [Error] Web Service Request Failed.";
                if (httpWebServiceInvoker.timeout()) { debugLog += " Reason: Timeout"; }
                else {
                    debugLog += " HTTP Status: " + httpWebServiceInvoker.responseStatusCode();
                    debugLog += " Description: " + httpWebServiceInvoker.resultDescription();
                    try { debugLog += " Server Response: " + httpWebServiceInvoker.responseBody(); } catch(e) { /* ignore */}
                }
            }
        } catch (e) {
            debugLog += "\n [Error] Exception during Web Service communication: " + e;
        }

        // 3e. 폴백(Fallback) 및 액션 값 클리핑
        if (!communicationSuccess) {
            debugLog += "\n Using fallback 0.0 U/h due to communication failure or invalid response.";
            agentDecidedAction_U_per_Hour = 0.0; // 통신 실패 시 기본 액션
        }

        var clippedAction_U_per_Hour = Math.max(ACTION_MIN_U_PER_H, Math.min(ACTION_MAX_U_PER_H, agentDecidedAction_U_per_Hour));
        if (clippedAction_U_per_Hour !== agentDecidedAction_U_per_Hour && communicationSuccess) {
            debugLog += "\n Action clipped from " + agentDecidedAction_U_per_Hour.toFixed(4) + " to " + clippedAction_U_per_Hour.toFixed(4) + " U/h.";
        }
        
        // 3f. 다음 5분 동안 적용할 인슐린 주입률 업데이트 및 이력용 값 저장
        current_rate_for_minute_application_U_per_H = clippedAction_U_per_Hour;
        lastAppliedAction_UperH = clippedAction_U_per_Hour; // 다음 5분 간격의 이력에 사용될 값

        debugLog += "\n Set current_rate_for_minute_application_U_per_H for next 5 mins to: " + current_rate_for_minute_application_U_per_H.toFixed(4) + " U/h.";
        debugLog += "\n--- End of 5-Minute Interval Logic ---";

    } // End of 5-minute interval block

    // debugLog += "\n[RunIteration] End of iteration " + currentMinuteAbs; // 분당 로그 너무 많으면 주석처리
}

// 에이전트에 전달할 상태 객체를 구성한다.
// bgHist와 insHist는 각각 최근 12개의 혈당값과 인슐린 주입률을 의미한다.
function prepareAgentState(bgHist, insHist, currentHour, timeToNextMealSteps, timeObj, mealObj, exerciseObj) {
    var historyForAgent = [];
    // bgHist, insHist는 이미 FEATURE_HISTORY_LENGTH 길이로 가정 (호출부에서 slice 등으로 복사 및 길이 관리)
    // 또는 여기서 길이를 한 번 더 보장할 수 있음
    for(var i = 0; i < FEATURE_HISTORY_LENGTH; i++){
        var bg = (i < bgHist.length && typeof bgHist[i] === 'number' && isFinite(bgHist[i])) ? bgHist[i] : 0.0;
        var ins = (i < insHist.length && typeof insHist[i] === 'number' && isFinite(insHist[i])) ? insHist[i] : 0.0;
        historyForAgent.push([bg, ins]);
    }

    var state = {
        "history": historyForAgent,
        "hour": currentHour,       // 현재 시간 (0-23)
        "meal": timeToNextMealSteps // 다음 식사까지 남은 시간 (5분 스텝)
    };

    if (USE_FEAT_VECTOR) { // 현재 명세서에서는 hour, meal을 직접 사용하므로 이 부분은 선택적
        var features = [];
        // 예시: features.push(timeObj.minutesPastMidnight / 1440.0); // G2P2C는 'hour'를 직접 사용
        // features.push(mealObj.minutesUntilNextMeal / 5.0 / 20.0); // 남은 시간 정규화 예시
        // ... 기타 필요한 특징 벡터 계산
        state["feat"] = features.map(function(f) { return isFinite(f) ? f : 0.0; });
    }
    return state;
}

// 시뮬레이션 종료 시 호출되어 모든 전역 상태를 초기화한다.
function cleanup() {
    debugLog = "[Cleanup] Simulation finished.";
    bgHistory = [];
    insulinActionHistory = [];
    isInitialized = false;
    lastAppliedAction_UperH = 0.0;
    current_rate_for_minute_application_U_per_H = 0.0;
    try {
        var success = httpWebServiceInvoker.performPostRequest(
            AGENT_API_URL + "/episode_end",
            "{}",
            "application/json",
            WEB_REQUEST_TIMEOUT_MS
        );
        if (!success) {
            debugLog += "\n[Cleanup] Failed to notify /episode_end.";
        }
    } catch(e) {
        debugLog += "\n[Cleanup] Exception during /episode_end POST: " + e;
    }
    // debugLog += "\n[Cleanup] Globals reset."; // 필요시 로그 추가
}

function requiresSqInsulinSupport() { return true; }
function requiresSqGlucoseSupport() { return false; }
function requiresInsulinDosingSupport() { return false; }
function requiresCPeptideSupport() { return false; }
function requiresSqGlucagonSupport() { return false; }
function debugLoggingEnabled() { return true; }
