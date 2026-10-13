# Data audit: data\clean\car-damage-v2

| | train | val | test | total |
|---|---|---|---|---|
| images | 1976 | 424 | 427 | 2827 |
| boxes | 3162 | 684 | 681 | 4527 |

## Class distribution (instances)

| class | train | val | test | total |
|---|---|---|---|---|
| doorouter-dent | 537 | 116 | 116 | 769 |
| front-bumper-dent | 531 | 114 | 114 | 759 |
| bonnet-dent | 420 | 90 | 90 | 600 |
| fender-dent | 347 | 74 | 74 | 495 |
| rear-bumper-dent | 283 | 61 | 61 | 405 |
| quaterpanel-dent | 245 | 53 | 53 | 351 |
| Headlight-Damage | 185 | 40 | 40 | 265 |
| Rear-windscreen-Damage | 122 | 27 | 26 | 175 |
| roof-dent | 92 | 20 | 20 | 132 |
| Front-Windscreen-Damage | 89 | 19 | 19 | 127 |
| RunningBoard-Dent | 80 | 18 | 17 | 115 |
| Taillight-Damage | 75 | 17 | 16 | 108 |
| boot-dent | 66 | 14 | 15 | 95 |
| Sidemirror-Damage | 56 | 12 | 13 | 81 |
| pillar-dent | 19 | 5 | 4 | 28 |
| Signlight-Damage | 15 | 4 | 3 | 22 |

Imbalance ratio (largest / smallest class): **35.0**

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

- Unique image sizes: 566; most common: [['259x194', 452], ['275x183', 199], ['1600x1200', 194], ['300x168', 96], ['800x533', 96]]
- Aspect ratio (w/h): {'min': 0.46, 'median': 1.34, 'max': 3.39}
- Box area / image area: {'p05': 0.0195, 'median': 0.1246, 'p95': 0.6238, 'tiny_share': 0.0}
- Boxes per image (mean): 1.6
