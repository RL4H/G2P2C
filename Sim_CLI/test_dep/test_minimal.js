// 파일명: test_minimal.js
// 목적: DMMS.R 플러그인 로딩 및 기본 실행, 로깅 기능 테스트용 최소 코드

// --- 전역 변수 ---
var isInitialized = false;
var debugLog = "";

// --- DMMS.R 필수 함수 ---

// 입력 신호를 하나 받는다고 가정 (XML 설정과 맞추기 위해)
function numSignals() { return 1; }

// 해당 입력 신호에 대한 설명
function signalDescription(signalIndex) {
    if (signalIndex === 0) { return "Dummy Input Signal (e.g., CGM)"; }
    return "Unknown Signal";
}

// 초기화 함수: 간단한 로그만 남기고 초기화 플래그 설정
function initialize(popName, subjName, simDuration, configDir, baseResultsDir, simName) {
    debugLog = "[MinimalTest Initialize] Script loaded for subject: " + subjName;
    try {
        isInitialized = true;
        debugLog += "\n[MinimalTest Initialize] isInitialized = true";
    } catch (e) {
        debugLog += "\n[MinimalTest Initialize] Error during minimal init: " + e;
        isInitialized = false;
    }
}

// 반복 실행 함수: 초기화되었는지 확인하고 간단한 로그만 남김
function runIteration(subjObject, sensorSigArray, nextMealObject, nextExerciseObject, timeObject, modelInputsToModObject) {
    if (!isInitialized) {
        // 초기화 실패 시 반복 로그 방지 (필요시 추가)
        if (debugLog.indexOf("Minimal RunIteration called before initialize!") === -1) {
            debugLog = "[MinimalTest Error] Minimal RunIteration called before initialize!";
        }
        return;
    }

    // 현재 시뮬레이션 시간 가져오기 (오류 방지하며)
    var currentMinuteStr = "Minute NaN";
    if (timeObject && timeObject.minutesPastSimStart !== undefined && isFinite(timeObject.minutesPastSimStart)) {
        currentMinuteStr = "Minute " + timeObject.minutesPastSimStart;
    }

    // 간단한 실행 로그 남기기 (매 스텝 기록 확인용)
    // 주의: 매 스텝 로그를 남기면 파일이 커질 수 있으므로, 확인 후에는 주석 처리하거나 빈 문자열로 둘 수 있습니다.
    debugLog = "[MinimalTest RunIteration] Running at " + currentMinuteStr;

    // modelInputsToModObject에 대한 최소한의 작업 (필요하다면)
    // 예: modelInputsToModObject["sqInsulinNormalBolus"] = 0; // 값을 덮어쓰지 않도록 주의
}

// 정리 함수: 간단한 종료 로그
function cleanup() {
    debugLog = "[MinimalTest Cleanup] Simulation finished.";
    isInitialized = false;
}

// 유효성 검사 함수들 (기본값 반환)
function requiresSqInsulinSupport() { return false; } // 여기서는 사용자 인슐린 안 씀
function requiresSqGlucoseSupport() { return false; }
function requiresInsulinDosingSupport() { return false; }
function requiresCPeptideSupport() { return false; }
function requiresSqGlucagonSupport() { return false; }

// 로그 활성화 함수 (가장 중요!)
function debugLoggingEnabled() { return true; }