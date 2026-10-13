# Data audit: data\raw\car-damage-v2

| | train | valid | test | total |
|---|---|---|---|---|
| images | 2301 | 678 | 79 | 3058 |
| boxes | 3679 | 1055 | 13 | 4747 |

## Class distribution (instances)

| class | train | valid | test | total |
|---|---|---|---|---|
| front-bumper-dent | 617 | 195 | 0 | 812 |
| doorouter-dent | 648 | 143 | 0 | 791 |
| bonnet-dent | 470 | 152 | 0 | 622 |
| fender-dent | 398 | 111 | 0 | 509 |
| rear-bumper-dent | 341 | 100 | 0 | 441 |
| quaterpanel-dent | 302 | 60 | 0 | 362 |
| Headlight-Damage | 210 | 68 | 3 | 281 |
| Rear-windscreen-Damage | 131 | 51 | 1 | 183 |
| roof-dent | 112 | 25 | 0 | 137 |
| Front-Windscreen-Damage | 97 | 33 | 0 | 130 |
| Taillight-Damage | 89 | 31 | 2 | 122 |
| RunningBoard-Dent | 87 | 30 | 4 | 121 |
| boot-dent | 75 | 20 | 0 | 95 |
| Sidemirror-Damage | 61 | 26 | 0 | 87 |
| pillar-dent | 23 | 5 | 0 | 28 |
| Signlight-Damage | 17 | 5 | 3 | 25 |
| Bodypanel-Dent | 1 | 0 | 0 | 1 |

Imbalance ratio (largest / smallest class): **812.0**

## Issues

- Image issues: {'empty_label': 70}
- Box issues: {'degenerate': 3, 'duplicate': 1}
- Orphan label files: 0

## Duplicates

- near duplicate pairs: 187
- groups: 152
- images in groups: 322
- redundant images: 170
- cross split groups: 65
- groups with label conflicts: 44

## Images and boxes

- Unique image sizes: 595; most common: [['259x194', 491], ['275x183', 223], ['1600x1200', 194], ['300x168', 104], ['800x533', 100]]
- Aspect ratio (w/h): {'min': 0.46, 'median': 1.35, 'max': 3.39}
- Box area / image area: {'p05': 0.0197, 'median': 0.1308, 'p95': 0.6346, 'tiny_share': 0.0006}
- Boxes per image (mean): 1.55
