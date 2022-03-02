# TODO list

- [X] Show test images: GT, x, out
- [X] train&test on 2 datasets
- [X] split jpcl train/test dataset
- [ ] adversarial attack
- [X] modify network structures
- [X] data prepocessing
- [ ] tensorboard?

## issues:

loss 0.4---

evaluation remains same and very low

## Current results with no attack

### JPCL

| epoch | DIC       | JSC       |
| ----- | --------- | --------- |
| 50    | 0.9759810 | 0.9533155 |

### ISBI

| epoch | DIC       | JSC       |
| ----- | --------- | --------- |
| 10    | 0.7460034 | 0.6254935 |
| 20    | 0.7977703 | 0.6916992 |
| 40    | 0.7479041 | 0.6396641 |
| 60    | 0.7769870 | 0.6743310 |
| 80    | 0.7552193 | 0.6525360 |
| 100   | 0.7599165 | 0.6583515 |
| 120   | 0.7705888 | 0.6655517 |
| 140   | 0.7501323 | 0.6444697 |
| 160   | 0.7512601 | 0.6467024 |
| 180   | 0.7355833 | 0.6305872 |

### JPCL

network v2

| epoch | DIC       | JSC       |
| ----- | --------- | --------- |
| 50    | 0.9645614 | 0.9320910 |
| 100   | 0.9710603 | 0.9439933 |
| 250   | 0.9734785 | 0.9485437 |
| 300   | 0.9729999 | 0.9476723 |
