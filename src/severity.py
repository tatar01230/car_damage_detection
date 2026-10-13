"""Rule-based damage severity: minor / moderate / severe.

These are hand-written HEURISTICS, not learned and not validated against real repair costs or
claim outcomes. They exist to turn detections into a readable claim-style summary.

Rules
  1. Base level by damaged part (BASE_SEVERITY):
       - body-panel dents (door, fender, bumpers, bonnet, boot, quarter panel, running board): minor
       - lights, side mirror, rear windscreen: moderate (part replacement; lights and mirrors
         are needed for road legality)
       - roof dent: moderate (labor-intensive repair, can indicate a rollover)
       - front windscreen: severe (driver visibility; must be fixed before driving)
       - pillar dent: severe (structural pillar, occupant safety)
  2. Size: a damage whose box covers >= LARGE_AREA of the image is raised one level.
     0.30 is about the 75th percentile of box area in the training set (median 0.12).
     Caveat: area relative to the image depends on camera distance; a close-up of a small dent
     looks "large". This is the weakest rule.
  3. Overall severity = the most severe single damage, raised one level if there are
     >= MANY_DAMAGES detections (14.5% of training images have 3 or more boxes).
     No detections -> "none".
"""

LEVELS = ["minor", "moderate", "severe"]

BASE_SEVERITY = {
    "doorouter-dent": "minor",
    "fender-dent": "minor",
    "front-bumper-dent": "minor",
    "rear-bumper-dent": "minor",
    "bonnet-dent": "minor",
    "boot-dent": "minor",
    "quaterpanel-dent": "minor",
    "RunningBoard-Dent": "minor",
    "roof-dent": "moderate",
    "Headlight-Damage": "moderate",
    "Taillight-Damage": "moderate",
    "Signlight-Damage": "moderate",
    "Sidemirror-Damage": "moderate",
    "Rear-windscreen-Damage": "moderate",
    "Front-Windscreen-Damage": "severe",
    "pillar-dent": "severe",
}

LARGE_AREA = 0.30
MANY_DAMAGES = 3


def raise_level(level: str) -> str:
    return LEVELS[min(LEVELS.index(level) + 1, len(LEVELS) - 1)]


def damage_severity(damage_type: str, area_ratio: float) -> str:
    level = BASE_SEVERITY[damage_type]  # KeyError on purpose: every model class must have a rule
    return raise_level(level) if area_ratio >= LARGE_AREA else level


def overall_severity(severities: list[str]) -> str:
    if not severities:
        return "none"
    worst = max(severities, key=LEVELS.index)
    return raise_level(worst) if len(severities) >= MANY_DAMAGES else worst
