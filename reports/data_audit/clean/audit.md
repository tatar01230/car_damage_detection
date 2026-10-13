# Data audit: data\clean\car-damage-v2

| | train | val | test | total |
|---|---|---|---|---|
| images | 1993 | 421 | 407 | 2821 |
| boxes | 3148 | 681 | 681 | 4510 |

## Class distribution (instances)

| class | train | val | test | total |
|---|---|---|---|---|
| doorouter-dent | 534 | 115 | 115 | 764 |
| front-bumper-dent | 528 | 114 | 114 | 756 |
| bonnet-dent | 419 | 90 | 90 | 599 |
| fender-dent | 342 | 74 | 74 | 490 |
| rear-bumper-dent | 283 | 61 | 61 | 405 |
| quaterpanel-dent | 245 | 53 | 53 | 351 |
| Headlight-Damage | 185 | 40 | 40 | 265 |
| Rear-windscreen-Damage | 122 | 27 | 26 | 175 |
| roof-dent | 92 | 20 | 20 | 132 |
| Front-Windscreen-Damage | 89 | 19 | 19 | 127 |
| RunningBoard-Dent | 78 | 17 | 17 | 112 |
| Taillight-Damage | 75 | 16 | 17 | 108 |
| boot-dent | 66 | 15 | 14 | 95 |
| Sidemirror-Damage | 56 | 12 | 13 | 81 |
| pillar-dent | 19 | 4 | 5 | 28 |
| Signlight-Damage | 15 | 4 | 3 | 22 |

Imbalance ratio (largest / smallest class): **34.7**

## Issues

- Image issues: none
- Box issues: none
- Orphan label files: 0

## Duplicates

- near duplicate pairs: 0
- groups: 0
- images in groups: 0
- redundant images: 0
- cross split groups: 0
- groups with label conflicts: 0

## Images and boxes

- Unique image sizes: 566; most common: [['259x194', 452], ['275x183', 199], ['1600x1200', 194], ['300x168', 96], ['800x533', 91]]
- Aspect ratio (w/h): {'min': 0.46, 'median': 1.34, 'max': 3.39}
- Box area / image area: {'p05': 0.0195, 'median': 0.1256, 'p95': 0.6239, 'tiny_share': 0.0}
- Boxes per image (mean): 1.6
