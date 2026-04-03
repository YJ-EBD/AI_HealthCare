from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SURVEY_GROUPS: list[dict[str, Any]] = [
    {
        "key": "soyang",
        "label": "소양인",
        "subtitle": "열이 쉽게 오르고 추진력이 강한 경향",
        "items": [
            "성격이 급하고 추진력이 강함",
            "더위에 약하고 땀이 많음",
            "얼굴·상체에 열이 잘 오름",
            "잠이 얕고 예민함",
        ],
    },
    {
        "key": "taeeum",
        "label": "태음인",
        "subtitle": "체격이 크고 느긋하며 순환 정체가 올 수 있는 경향",
        "items": [
            "체격이 크거나 살이 잘 찜",
            "참을성이 많고 느긋함",
            "땀을 흘리면 개운함",
            "호흡기·피로·비만 경향",
        ],
    },
    {
        "key": "soeum",
        "label": "소음인",
        "subtitle": "몸이 차고 소화가 약하며 신중한 경향",
        "items": [
            "손발이 차고 추위를 탐",
            "소화가 약하고 설사·복통",
            "신중하고 걱정이 많음",
            "적게 먹어도 배부름",
        ],
    },
    {
        "key": "taeyang",
        "label": "태양인",
        "subtitle": "매우 희귀하며 상체 발달과 독립성이 강한 경향",
        "items": [
            "가슴·어깨 발달",
            "하체·복부 약함",
            "리더형·독립적",
            "소변·피로 문제",
        ],
    },
]


CONSTITUTION_LABELS = {group["key"]: group["label"] for group in SURVEY_GROUPS}


PROFILE_LIBRARY: dict[str, dict[str, Any]] = {
    "heat_soyang": {
        "label": "열소양형 (熱少陽型)",
        "summary": "열이 빠르게 오르고 피지·자극 반응이 커서 진정 중심 관리가 필요한 타입",
        "features": [
            "얼굴 홍조, 여드름, 염증성 트러블",
            "피지 분비 많음",
            "자극 후 악화 잘 됨",
        ],
        "avoid": [
            "고출력 RF",
            "장시간 온열",
            "강한 EMS",
        ],
        "settings": [
            "LED: 청색 / 녹색",
            "온열: OFF 또는 ≤ 37~38℃",
            "진동: 저강도",
            "쿨링 기능 적극 사용",
        ],
        "goal": "열 진정 + 염증 억제",
    },
    "weak_soyang": {
        "label": "허소양형 (虛少陽型)",
        "summary": "겉열은 있지만 회복력이 떨어져 자극량을 세밀하게 조절해야 하는 타입",
        "features": [
            "겉은 열, 속은 기허",
            "트러블은 잦으나 회복 느림",
            "피부 얇고 예민",
        ],
        "avoid": [
            "강한 냉각",
            "장시간 고출력",
        ],
        "settings": [
            "LED: 녹색 + 적색 혼합",
            "온열: 약온 (38~39℃)",
            "EMS: 미세 순환 레벨",
            "시간: 짧고 자주",
        ],
        "goal": "열 완화 + 회복력 보강",
    },
    "damp_taeeum": {
        "label": "담태음형 (痰太陰型)",
        "summary": "부종과 처짐, 순환 정체 경향이 있어 배출과 탄력 중심 관리가 필요한 타입",
        "features": [
            "얼굴 부종, 처짐",
            "모공 확장, 유분 많음",
            "림프 정체",
        ],
        "avoid": [
            "단순 보습 위주 관리",
            "저자극만 하는 관리",
        ],
        "settings": [
            "EMS: 림프 배출 모드",
            "진동: 중~강",
            "온열: 40℃ 내외",
            "흡입 마사지 가능",
        ],
        "goal": "배출 + 순환 + 탄력",
    },
    "dry_taeeum": {
        "label": "건태음형 (乾太陰型)",
        "summary": "태음 경향이지만 건조와 잔주름이 두드러져 보습과 탄력 유지가 필요한 타입",
        "features": [
            "태음인이지만 마른 편",
            "건조, 잔주름",
            "순환은 나쁘지 않음",
        ],
        "avoid": [
            "강한 배출 위주 관리",
            "과도한 EMS",
        ],
        "settings": [
            "LED: 적색 / NIR",
            "온열: 40~41℃",
            "진동: 중간",
            "보습 앰플 병행",
        ],
        "goal": "탄력 + 보습 + 혈류 유지",
    },
    "cold_soeum": {
        "label": "냉소음형 (冷少陰型)",
        "summary": "차가움과 혈색 부족이 중심이라 온열과 혈류 개선이 핵심인 타입",
        "features": [
            "손발·얼굴 차가움",
            "안색 창백",
            "피부 혈색 부족",
        ],
        "avoid": [
            "냉각, 쿨링",
            "차가운 젤",
        ],
        "settings": [
            "온열: 41~42℃ (핵심)",
            "LED: 적색",
            "저주파: 혈류 모드",
            "시술 시간 충분히",
        ],
        "goal": "온보(溫補) + 혈색 개선",
    },
    "weak_soeum": {
        "label": "허소음형 (虛少陰型)",
        "summary": "회복력이 매우 낮아 에너지 보존과 재생 위주 접근이 필요한 타입",
        "features": [
            "극심한 피로형",
            "피부 얇고 처짐",
            "회복력 매우 낮음",
        ],
        "avoid": [
            "강한 EMS",
            "장시간 시술",
        ],
        "settings": [
            "온열: 39~40℃",
            "LED: 적색 + 근적외",
            "EMS: OFF 또는 최저",
            "시술 간격 길게",
        ],
        "goal": "에너지 보존 + 재생",
    },
    "solid_taeyang": {
        "label": "실태양형 (實太陽型)",
        "summary": "열과 건조, 예민성이 겹쳐 자극 최소화와 열 분산이 중요한 희귀 타입",
        "features": [
            "상체 발달",
            "열 많고 건조",
            "예민한 피부",
        ],
        "avoid": [
            "열 자극",
            "장시간 접촉",
        ],
        "settings": [
            "LED: 청·녹",
            "냉각 위주",
            "접촉 최소화",
        ],
        "goal": "열 분산 + 자극 최소",
    },
    "weak_taeyang": {
        "label": "허태양형 (虛太陽型)",
        "summary": "건조와 예민함, 기력 저하가 겹쳐 안정적인 보조가 필요한 희귀 타입",
        "features": [
            "기력 부족",
            "건조 + 예민",
            "상열하허",
        ],
        "avoid": [
            "강한 냉각",
            "강자극",
        ],
        "settings": [
            "LED: 적색",
            "온열: 저온·단시간",
            "진동 최소",
        ],
        "goal": "기혈 보조 + 안정",
    },
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def extract_signal_features(psl_report: dict[str, Any] | None, face_result: dict[str, Any] | None) -> dict[str, float]:
    heart = (psl_report or {}).get("heart_rate") or {}
    hrv = (psl_report or {}).get("hrv") or {}
    stress = (psl_report or {}).get("stress") or {}
    circulation = (psl_report or {}).get("circulation") or {}
    vascular_health = (psl_report or {}).get("vascular_health") or {}
    vascular_age = (psl_report or {}).get("vascular_age") or {}
    blood_pressure = (psl_report or {}).get("blood_pressure") or {}
    meta = (psl_report or {}).get("metadata") or {}
    metrics = (face_result or {}).get("metrics") or {}

    def face_metric(metric_key: str) -> float:
        return safe_float((metrics.get(metric_key) or {}).get("score"))

    return {
        "hr": safe_float(heart.get("heart_rate_bpm")),
        "rmssd": safe_float(hrv.get("rmssd_ms")),
        "sdnn": safe_float(hrv.get("sdnn_ms")),
        "stress": safe_float(stress.get("stress_score")),
        "circulation": safe_float(circulation.get("circulation_score")),
        "perfusion": safe_float(circulation.get("perfusion_index")),
        "vascular_health": safe_float(vascular_health.get("vascular_health_score")),
        "vascular_age_gap": safe_float(vascular_age.get("vascular_age_gap")),
        "sbp": safe_float(blood_pressure.get("estimated_sbp")),
        "dbp": safe_float(blood_pressure.get("estimated_dbp")),
        "signal_quality": safe_float(meta.get("signal_quality_score")),
        "overall_face": safe_float((face_result or {}).get("overall_score")),
        "wrinkle": face_metric("wrinkle"),
        "pigmentation": face_metric("pigmentation"),
        "pore": face_metric("pore"),
        "dryness": face_metric("dryness"),
        "sagging": face_metric("sagging"),
    }


def compute_constitution_scores(survey_answers: dict[str, int] | None) -> dict[str, int]:
    answers = survey_answers or {}
    return {group["key"]: max(0, safe_int(answers.get(group["key"]), 0)) for group in SURVEY_GROUPS}


def determine_constitution(survey_answers: dict[str, int] | None, features: dict[str, float]) -> tuple[str, str, list[str]]:
    scores = compute_constitution_scores(survey_answers)
    top_score = max(scores.values()) if scores else 0
    candidates = [key for key, value in scores.items() if value == top_score]

    evidence: list[str] = [f"설문 최고 점수: {', '.join(CONSTITUTION_LABELS[key] for key in candidates)}"]
    if len(candidates) == 1:
        selected = candidates[0]
        return selected, CONSTITUTION_LABELS[selected], evidence

    weighted_scores = dict(scores)
    if "soeum" in weighted_scores:
        weighted_scores["soeum"] += int(features["circulation"] <= 35.0) + int(features["hr"] <= 68.0) + int(features["dryness"] >= 45.0)
    if "soyang" in weighted_scores:
        weighted_scores["soyang"] += int(features["stress"] >= 55.0) + int(features["hr"] >= 82.0) + int(features["pore"] >= 45.0)
    if "taeeum" in weighted_scores:
        weighted_scores["taeeum"] += int(features["sagging"] >= 45.0) + int(features["pore"] >= 50.0) + int(features["vascular_age_gap"] >= 0.0)
    if "taeyang" in weighted_scores:
        weighted_scores["taeyang"] += int(features["stress"] >= 60.0) + int(features["dryness"] >= 50.0)

    selected = max(candidates, key=lambda key: weighted_scores.get(key, 0))
    evidence.append(f"동점 보정 후 선택: {CONSTITUTION_LABELS[selected]}")
    return selected, CONSTITUTION_LABELS[selected], evidence


def build_profile_recommendation(
    survey_answers: dict[str, int] | None,
    psl_report: dict[str, Any] | None,
    face_result: dict[str, Any] | None,
    survey_details: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    features = extract_signal_features(psl_report, face_result)
    constitution_key, constitution_label, evidence = determine_constitution(survey_answers, features)
    scores = compute_constitution_scores(survey_answers)
    detail_map = survey_details or {}

    heat_signal = sum(
        [
            features["stress"] >= 55.0,
            features["hr"] >= 82.0,
            features["sbp"] >= 125.0,
            features["pore"] >= 50.0,
            features["pigmentation"] >= 50.0,
        ]
    )
    deficiency_signal = sum(
        [
            features["rmssd"] <= 24.0,
            features["circulation"] <= 35.0,
            features["vascular_health"] <= 45.0,
            features["dryness"] >= 50.0,
            features["sagging"] >= 50.0,
        ]
    )
    damp_signal = sum(
        [
            features["pore"] >= 50.0,
            features["sagging"] >= 50.0,
            features["circulation"] <= 45.0,
            features["vascular_age_gap"] >= 0.0,
        ]
    )
    dry_signal = sum(
        [
            features["dryness"] >= 50.0,
            features["wrinkle"] >= 45.0,
            features["pore"] <= 40.0,
        ]
    )
    cold_signal = sum(
        [
            features["circulation"] <= 32.0,
            features["hr"] <= 66.0,
            features["sbp"] <= 108.0,
            features["dbp"] <= 68.0,
        ]
    )

    profile_key = "heat_soyang"
    if constitution_key == "soyang":
        profile_key = "heat_soyang" if heat_signal >= deficiency_signal else "weak_soyang"
        evidence.append(f"열 신호 {heat_signal}, 허약 신호 {deficiency_signal}")
    elif constitution_key == "taeeum":
        profile_key = "damp_taeeum" if damp_signal >= dry_signal else "dry_taeeum"
        evidence.append(f"정체 신호 {damp_signal}, 건조 신호 {dry_signal}")
    elif constitution_key == "soeum":
        profile_key = "weak_soeum" if deficiency_signal >= 3 and dry_signal >= 1 else "cold_soeum"
        evidence.append(f"냉 신호 {cold_signal}, 허약 신호 {deficiency_signal}")
        if profile_key == "cold_soeum" and cold_signal < 2 and deficiency_signal >= 3:
            profile_key = "weak_soeum"
    elif constitution_key == "taeyang":
        profile_key = "solid_taeyang" if heat_signal >= deficiency_signal else "weak_taeyang"
        evidence.append(f"열 신호 {heat_signal}, 허약 신호 {deficiency_signal}")

    profile_meta = PROFILE_LIBRARY[profile_key]
    for group in SURVEY_GROUPS:
        group_key = group["key"]
        selected_items = [str(item) for item in detail_map.get(group_key, []) if str(item).strip()]
        if selected_items:
            evidence.append(f"{group['label']} 체크 {scores.get(group_key, len(selected_items))}개: {', '.join(selected_items)}")
        else:
            evidence.append(f"{group['label']} 체크 {scores.get(group_key, 0)}개")
    evidence.extend(
        [
            f"스트레스 {features['stress']:.1f}, 심박수 {features['hr']:.1f}, 순환 {features['circulation']:.1f}",
            f"주름 {features['wrinkle']:.1f}, 색소 {features['pigmentation']:.1f}, 모공 {features['pore']:.1f}, 건조 {features['dryness']:.1f}, 처짐 {features['sagging']:.1f}",
        ]
    )

    return {
        "constitution_key": constitution_key,
        "constitution_label": constitution_label,
        "survey_scores": scores,
        "survey_details": detail_map,
        "profile_key": profile_key,
        "profile_label": profile_meta["label"],
        "summary": profile_meta["summary"],
        "features": list(profile_meta["features"]),
        "avoid": list(profile_meta["avoid"]),
        "settings": list(profile_meta["settings"]),
        "goal": profile_meta["goal"],
        "evidence": evidence,
    }


def format_survey_summary(
    survey_answers: dict[str, int] | None,
    survey_details: dict[str, list[str]] | None = None,
) -> str:
    scores = compute_constitution_scores(survey_answers)
    detail_map = survey_details or {}
    lines = ["체질 설문 요약", "============"]
    if not any(scores.values()):
        lines.append("아직 선택된 문항이 없습니다.")
    for group in SURVEY_GROUPS:
        key = group["key"]
        score = scores.get(key, 0)
        selected_items = [str(item) for item in detail_map.get(key, []) if str(item).strip()]
        lines.append(f"{group['label']}: {score}개 선택")
        if selected_items:
            lines.extend(f"  - {item}" for item in selected_items)
        else:
            lines.append("  - 선택 없음")
    return "\n".join(lines)


def format_profile_recommendation(recommendation: dict[str, Any] | None) -> str:
    if not recommendation:
        return "아직 체질 기반 추천 결과가 없습니다."

    survey_details = recommendation.get("survey_details") or {}
    lines = [
        f"기본 체질: {recommendation['constitution_label']}",
        f"판정 유형: {recommendation['profile_label']}",
        f"요약: {recommendation['summary']}",
        "",
        "설문 반영",
    ]
    for group in SURVEY_GROUPS:
        selected_items = [str(item) for item in survey_details.get(group["key"], []) if str(item).strip()]
        if selected_items:
            lines.append(f"- {group['label']}: {len(selected_items)}개 선택 ({', '.join(selected_items)})")
        else:
            lines.append(f"- {group['label']}: 선택 없음")
    lines.extend(
        [
            "",
        "특징",
        ]
    )
    lines.extend(f"- {item}" for item in recommendation["features"])
    lines.append("")
    lines.append("피해야 할 것")
    lines.extend(f"- {item}" for item in recommendation["avoid"])
    lines.append("")
    lines.append("권장 기기 세팅")
    lines.extend(f"- {item}" for item in recommendation["settings"])
    lines.append("")
    lines.append(f"목표: {recommendation['goal']}")
    lines.append("")
    lines.append("판정 근거")
    lines.extend(f"- {item}" for item in recommendation["evidence"])
    return "\n".join(lines)
