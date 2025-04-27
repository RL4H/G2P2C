// DMMS.R JavaScript Plugin Control Element for RL Agent Interface
// Version: 0.4.1 (Modification Plan 1 Applied)
// Changes: Simplified initialize, moved array management to runIteration.
// Assumes Python FastAPI server on the agent side.

// --- 설정 변수 ---
var AGENT_COMMUNICATION_METHOD = 'WEB_SERVICE'; // 웹 서비스 방식 사용
var AGENT_API_URL = "http://127.0.0.1:5000"; // FastAPI 서버 주소 - ** 실제 주소로 수정 필요 **
var WEB_REQUEST_TIMEOUT_MS = 5000; // 웹 서비스 타임아웃 (밀리초) - 필요시 조정

var INSULIN_NAME_DEFINED_IN_DMMS = "PPO_Agent_Insulin"; // DMMS.R 'Insulin Definitions'에서 정의한 이름 - ** 확인 필요 **
var INSULIN_TARGET_KEY = "sqCustomInsulin1"; // ** 사용자가 정의한 순서에 따라 sqCustomInsulin2 등이 될 수 있음. 확인 필요! **
var FALLBACK_INSULIN_KEY = "sqInsulinNormalBolus"; // sqCustomInsulinX 키 실패 시 사용할 표준 볼루스 키

var PMOL_PER_UNIT = 6000.0;
var FEATURE_HISTORY_LENGTH = 12;
var ACTION_MIN_U_PER_H = 0.0;
var ACTION_MAX_U_PER_H = 5.0;
var USE_FEAT_VECTOR = false; // 추가 특징 벡터 사용 여부 - ** 결정 및 구현 필요 **

// --- 전역 변수 (상태 관리용) ---
var bgHistory = []; // 선언만 하고 초기화는 runIteration에서
var insulinActionHistory = []; // 선언만 하고 초기화는 runIteration에서
var isInitialized = false;
var lastAppliedAction_UperH = 0.0;
var debugLog = ""; // 맨 위로 이동

// --- DMMS.R 필수 함수 ---

function numSignals() { return 1; } // CGM 입력 하나

function signalDescription(signalIndex) {
    if (signalIndex === 0) { return "Blood Glucose Input (CGM)"; }
    return "Unknown Signal";
}

// 수정된 initialize 함수 (단순화)
function initialize(popName, subjName, simDuration, configDir, baseResultsDir, simName) {
    debugLog = "[Initialize] Attempting simple initialization for subject: " + subjName + ", Sim: " + simName + " starting (WebService Mode).";
    try {
        // 배열 초기화 로직 제거!
        lastAppliedAction_UperH = 0.0; // 이전 액션 초기화
        isInitialized = true; // 플래그 설정만 수행
        debugLog += "\n[Initialize] Simple initialization complete. isInitialized set to true.";
    } catch (e) {
        debugLog += "\n[Initialize] !!! CRITICAL ERROR IN SIMPLE INITIALIZE: " + e;
        isInitialized = false; // 오류 발생 시 false 유지
    }
}

// 수정된 runIteration 함수 (배열 관리 로직 추가)
function runIteration(subjObject, sensorSigArray, nextMealObject, nextExerciseObject, timeObject, modelInputsToModObject) {
    // initialize 실패 시 실행 방지
    if (!isInitialized) {
        // 이 메시지는 initialize 실패 시 매번 기록될 것이므로, 최초 오류 외에는 반복 로깅 방지 가능
        if (debugLog.indexOf("runIteration called before initialize!") === -1) {
             debugLog = "[Error] runIteration called before initialize!";
        }
        return;
    }

    // runIteration 처음 호출 시 또는 배열 길이가 부족할 때 초기화/길이 조정
    // (unshift는 배열 앞에 추가하므로, 길이를 채울 땐 반복 push나 다른 방식이 나을 수 있음)
    // 여기서는 간단하게 push/shift 방식으로 길이 유지
    if (bgHistory.length < FEATURE_HISTORY_LENGTH) {
        debugLog += "\n[RunIteration] Initializing history arrays with 0.0 inside runIteration.";
        // FEATURE_HISTORY_LENGTH 길이만큼 0.0으로 채우기
        while(bgHistory.length < FEATURE_HISTORY_LENGTH) bgHistory.push(0.0);
        while(insulinActionHistory.length < FEATURE_HISTORY_LENGTH) insulinActionHistory.push(0.0);
        // 주의: 이 방식은 첫 실행 또는 리셋 시에만 동작하도록 개선할 수 있음
    }


    // 시간 정보 가져오기 (속성 이름 수정!)
    var currentMinute = NaN;
    var timeLog = "";
    if (timeObject && typeof timeObject === 'object') {
         // 속성 존재 여부 확인 후 사용
        if (timeObject.minutesPastSimStart !== undefined && isFinite(timeObject.minutesPastSimStart)) {
            currentMinute = timeObject.minutesPastSimStart;
            timeLog = "Minute: " + currentMinute + " (from minutesPastSimStart)";
        } else if (timeObject.dayOfYear !== undefined && isFinite(timeObject.dayOfYear) && timeObject.minutesPastMidnight !== undefined && isFinite(timeObject.minutesPastMidnight)) {
            currentMinute = (timeObject.dayOfYear * 1440 + timeObject.minutesPastMidnight); // 대체 계산
             timeLog = "Minute: " + currentMinute + " (calculated from DoY/MoM)";
        } else {
             timeLog = "Minute: NaN (Could not determine time from timeObject: " + JSON.stringify(timeObject) + ")";
        }
    } else {
         timeLog = "Minute: NaN (timeObject is invalid)";
    }
    debugLog = "[RunIteration] " + timeLog;


    // 1. 상태 정보 수집 및 가공
    var currentCGM = sensorSigArray[0];
    if (isNaN(currentCGM) || !isFinite(currentCGM)){
        debugLog += "\n [Warning] Invalid CGM value received: " + currentCGM + ". Using 0 for history.";
        currentCGM = 0.0; // 유효하지 않은 값은 0으로 처리 (또는 이전 값 유지 등 정책 결정)
    }
    debugLog += "\n Current CGM: " + currentCGM.toFixed(4);

    // 배열 길이 유지하며 값 업데이트 (push & shift)
    bgHistory.push(currentCGM);
    if (bgHistory.length > FEATURE_HISTORY_LENGTH) {
        bgHistory.shift(); // 가장 오래된 데이터 제거
    }
    insulinActionHistory.push(lastAppliedAction_UperH); // 이전 스텝에서 *적용된* 액션 추가
    if (insulinActionHistory.length > FEATURE_HISTORY_LENGTH) {
        insulinActionHistory.shift(); // 가장 오래된 데이터 제거
    }

    // prepareAgentState 호출 시 현재 배열 전달 (길이가 FEATURE_HISTORY_LENGTH가 되도록 함)
    // slice를 사용하면 원본 배열을 변경하지 않고 복사본을 만들 수 있음
    var currentBgHistory = bgHistory.slice(); // 현재 배열 복사
    var currentInsHistory = insulinActionHistory.slice(); // 현재 배열 복사

    // 길이가 부족한 경우 앞부분을 0으로 채움 (호출 전 확인)
    while (currentBgHistory.length < FEATURE_HISTORY_LENGTH) currentBgHistory.unshift(0.0);
    while (currentInsHistory.length < FEATURE_HISTORY_LENGTH) currentInsHistory.unshift(0.0);

    var agentState = prepareAgentState(currentBgHistory, currentInsHistory, timeObject, nextMealObject, nextExerciseObject);
    debugLog += "\n Prepared Agent State (History only): " + JSON.stringify(agentState); // 로그 간소화

    // 2. 에이전트와 통신 (웹 서비스 - HTTP POST)
    var agentAction_U_per_Hour = 0.0; // 기본값
    var communicationSuccess = false;

    try {
        var requestBody = JSON.stringify(agentState);
        var contentType = "application/json";
        var success = httpWebServiceInvoker.performPostRequest(AGENT_API_URL + "/predict_action", requestBody, contentType, WEB_REQUEST_TIMEOUT_MS); // 엔드포인트 추가

        if (success && !httpWebServiceInvoker.timeout() && httpWebServiceInvoker.responseStatusCode() >= 200 && httpWebServiceInvoker.responseStatusCode() < 300) {
            var responseBody = httpWebServiceInvoker.responseBody();
            debugLog += "\n Raw Response: " + responseBody;
            try {
                var parsedResponse = JSON.parse(responseBody);
                if (parsedResponse && typeof parsedResponse.insulin_action_U_per_h === 'number' && isFinite(parsedResponse.insulin_action_U_per_h)) {
                    agentAction_U_per_Hour = parsedResponse.insulin_action_U_per_h;
                    communicationSuccess = true;
                    debugLog += "\n Parsed Action (U/h): " + agentAction_U_per_Hour.toFixed(4);
                } else {
                    debugLog += "\n [Error] Invalid JSON structure/value in response: " + responseBody;
                }
            } catch (parseError) {
                debugLog += "\n [Error] Failed to parse JSON response: " + parseError + "\nResponse body: " + responseBody;
            }
        } else {
            debugLog += "\n [Error] Web Service Request Failed.";
            if (httpWebServiceInvoker.timeout()) { debugLog += " Reason: Timeout"; }
            else {
                debugLog += " HTTP Status: " + httpWebServiceInvoker.responseStatusCode();
                debugLog += " Description: " + httpWebServiceInvoker.resultDescription();
                try { debugLog += " Server Response: " + httpWebServiceInvoker.responseBody(); } catch(e) {}
            }
        }
    } catch (e) {
        debugLog += "\n [Error] Exception during Web Service communication: " + e;
    }

    // Fallback 적용 (통신 실패 또는 유효하지 않은 값)
    if (!communicationSuccess) {
        debugLog += "\n Using fallback 0.0 U/h due to communication failure or invalid response.";
        agentAction_U_per_Hour = 0.0;
    }

    // 3. 행동(Action) 적용
    // 액션 값 클리핑 (0~5 U/h)
    var clippedAction_U_per_Hour = Math.max(ACTION_MIN_U_PER_H, Math.min(ACTION_MAX_U_PER_H, agentAction_U_per_Hour));
    if (clippedAction_U_per_Hour !== agentAction_U_per_Hour && communicationSuccess) {
        debugLog += "\n Action clipped from " + agentAction_U_per_Hour.toFixed(4) + " to " + clippedAction_U_per_Hour.toFixed(4) + " U/h.";
    }

    // 단위 변환: U/h -> pmol/min
    var dose_pmol_per_Min = clippedAction_U_per_Hour / 60.0 * PMOL_PER_UNIT;
    debugLog += "\n Converted Dose (pmol/min): " + dose_pmol_per_Min.toFixed(4);

    // 시뮬레이터 입력 객체 업데이트 (Custom Insulin Key 우선 시도, Fallback 적용)
    var keyToUse = "";
    if (INSULIN_TARGET_KEY in modelInputsToModObject) {
        keyToUse = INSULIN_TARGET_KEY;
    } else if (FALLBACK_INSULIN_KEY in modelInputsToModObject) {
        keyToUse = FALLBACK_INSULIN_KEY;
        debugLog += "\n [Warning] Target key '" + INSULIN_TARGET_KEY + "' not found! Using fallback key '" + keyToUse + "'. Check INSULIN_TARGET_KEY setting.";
    } else {
        debugLog += "\n [Error] Neither target key '" + INSULIN_TARGET_KEY + "' nor fallback key '" + FALLBACK_INSULIN_KEY + "' found in modelInputsToModObject! Cannot apply insulin dose.";
        // 키를 찾을 수 없을 때의 처리 (예: 오류 상태 설정 또는 0 적용 등)
        // 사용 가능한 키 로깅 추가:
        debugLog += "\n Available keys in modelInputsToModObject: " + Object.keys(modelInputsToModObject).join(', ');
    }

    if (keyToUse !== "") {
        // 기존 값에 더하는 것이 아니라, 해당 스텝의 주입률로 설정해야 함
        modelInputsToModObject[keyToUse] = dose_pmol_per_Min;
        debugLog += "\n Applied dose to key: " + keyToUse + " = " + dose_pmol_per_Min.toFixed(4) + " pmol/min";
    }

    // 다음 스텝 이력 저장을 위해 현재 적용한 액션 저장 (U/h 단위, 클리핑된 값)
    lastAppliedAction_UperH = clippedAction_U_per_Hour;

    debugLog += "\n[RunIteration] End of iteration " + (isNaN(currentMinute) ? "NaN" : currentMinute);
}

// 수정된 prepareAgentState 함수 (입력 배열 길이 보장 가정)
function prepareAgentState(bgHist, insHist, timeObj, mealObj, exerciseObj) {
    var history = [];
    // 입력 배열(bgHist, insHist)의 길이가 FEATURE_HISTORY_LENGTH라고 가정
    for(var i = 0; i < FEATURE_HISTORY_LENGTH; i++){
         // 방어 코드 추가: 배열 요소 유효성 검사
        var bg = (i < bgHist.length && typeof bgHist[i] === 'number' && isFinite(bgHist[i])) ? bgHist[i] : 0.0;
        var ins = (i < insHist.length && typeof insHist[i] === 'number' && isFinite(insHist[i])) ? insHist[i] : 0.0;
        history.push([bg, ins]);
    }

    var state = { "history": history };

    if (USE_FEAT_VECTOR) {
        var features = [];
        // --- 특징 벡터 계산 로직 구현 ---
        // 예시: features.push(timeObj.minutesPastMidnight / 1440.0);
        // features.push(mealObj.minutesUntilNextMeal / 60.0); // 시간 단위로 정규화 등
        // features.push(exerciseObj.minutesUntilNextSession / 60.0);
        // -------------------------------
        // NaN/Inf 값 방지 필요
        state["feat"] = features.map(function(f) { return isFinite(f) ? f : 0.0; });
    }
    return state;
}


function cleanup() {
    debugLog = "[Cleanup] Simulation finished for subject."; // Subject 이름 추가 가능 (subjName은 전역 변수가 아님)
    // 전역 변수 초기화 (선택 사항)
    bgHistory = [];
    insulinActionHistory = [];
    isInitialized = false;
    lastAppliedAction_UperH = 0.0;
}


function requiresSqInsulinSupport() { return true; } // 사용자 정의 인슐린 사용 가정
function requiresSqGlucoseSupport() { return false; }
function requiresInsulinDosingSupport() { return false; } // DMMS 내부 dosing 로직 사용 안 함
function requiresCPeptideSupport() { return false; }
function requiresSqGlucagonSupport() { return false; }
function debugLoggingEnabled() { return true; } // 디버그 로그 활성화