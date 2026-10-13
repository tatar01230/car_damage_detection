# Data audit: data\raw\car-damage-v2

| | train | valid | test | total |
|---|---|---|---|---|
| images | 2296 | 677 | 78 | 3051 |
| boxes | 3664 | 1051 | 12 | 4727 |

## Class distribution (instances)

| class | train | valid | test | total |
|---|---|---|---|---|
| front-bumper-dent | 614 | 195 | 0 | 809 |
| doorouter-dent | 642 | 142 | 0 | 784 |
| bonnet-dent | 469 | 152 | 0 | 621 |
| fender-dent | 394 | 110 | 0 | 504 |
| rear-bumper-dent | 341 | 100 | 0 | 441 |
| quaterpanel-dent | 301 | 60 | 0 | 361 |
| Headlight-Damage | 210 | 68 | 3 | 281 |
| Rear-windscreen-Damage | 131 | 51 | 1 | 183 |
| roof-dent | 112 | 25 | 0 | 137 |
| Front-Windscreen-Damage | 97 | 33 | 0 | 130 |
| Taillight-Damage | 89 | 31 | 2 | 122 |
| RunningBoard-Dent | 87 | 28 | 3 | 118 |
| boot-dent | 75 | 20 | 0 | 95 |
| Sidemirror-Damage | 61 | 26 | 0 | 87 |
| pillar-dent | 23 | 5 | 0 | 28 |
| Signlight-Damage | 17 | 5 | 3 | 25 |
| Bodypanel-Dent | 1 | 0 | 0 | 1 |

Imbalance ratio (largest / smallest class): **809.0**

## Issues

- Image issues: {'empty_label': 70}
- Box issues: {'degenerate': 3, 'duplicate': 1}
- Orphan label files: 0

## Duplicates

- near duplicate pairs: 186
- groups: 151
- images in groups: 320
- redundant images: 169
- cross split groups: 65
- groups with label conflicts: 44

## Images and boxes

- Unique image sizes: 595; most common: [['259x194', 491], ['275x183', 223], ['1600x1200', 194], ['300x168', 104], ['800x533', 95]]
- Aspect ratio (w/h): {'min': 0.46, 'median': 1.35, 'max': 3.39}
- Box area / image area: {'p05': 0.0197, 'median': 0.1316, 'p95': 0.6356, 'tiny_share': 0.0006}
- Boxes per image (mean): 1.55
