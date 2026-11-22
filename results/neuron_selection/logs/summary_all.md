
## Neuron selection (2d43012f8b9d7644efb9f1267c17bb6c65043f21; 3940 unique current rows)


### Pruning

| Dataset | Strategy | P | k | Criterion | Test | Δ random | z |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FashionMNIST | hypersphere | — | 250 | random | 0.80661 ± 0.0010019 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | — | 500 | random | 0.83071 ± 0.0006565 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | — | 750 | random | 0.84174 ± 0.00050337 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | — | 1000 | random | 0.84883 ± 0.00056314 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | — | 1500 | random | 0.85695 ± 0.00043493 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | — | 2000 | random | 0.86232 ± 0.00067046 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | — | 3000 | random | 0.86949 ± 0.00062298 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | — | 4000 | random | 0.87364 ± 0.00056808 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | — | 6000 | random | 0.88002 ± 0.00063365 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | — | 8000 | random | 0.88376 ± 0.00048401 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | 2000 | 2000 | pool | 0.86232 ± 0.00067046 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | 2000 | 1000 | activity | 0.84742 ± 0.00048849 (n=10) | -0.00141 ± 0.00040564 (n=10) | -3.476 |
| FashionMNIST | hypersphere | 2000 | 1000 | magnitude | 0.85193 ± 0.00052389 (n=10) | 0.0031 ± 0.0006261 (n=10) | 4.9513 |
| FashionMNIST | hypersphere | 2000 | 1000 | obs | 0.8562 ± 0.00054752 (n=10) | 0.00737 ± 0.00057813 (n=10) | 12.748 |
| FashionMNIST | hypersphere | 2000 | 1000 | obs_oneshot | 0.85595 ± 0.00056711 (n=10) | 0.00712 ± 0.0006688 (n=10) | 10.646 |
| FashionMNIST | hypersphere | 2000 | 500 | activity | 0.8289 ± 0.00064739 (n=10) | -0.00181 ± 0.0010012 (n=10) | -1.8079 |
| FashionMNIST | hypersphere | 2000 | 500 | magnitude | 0.83583 ± 0.0005149 (n=10) | 0.00512 ± 0.00043482 (n=10) | 11.775 |
| FashionMNIST | hypersphere | 2000 | 500 | obs | 0.84573 ± 0.00056471 (n=10) | 0.01502 ± 0.00070155 (n=10) | 21.41 |
| FashionMNIST | hypersphere | 2000 | 500 | obs_oneshot | 0.84178 ± 0.0006031 (n=10) | 0.01107 ± 0.001024 (n=10) | 10.81 |
| FashionMNIST | hypersphere | 2000 | 250 | activity | 0.80914 ± 0.00080142 (n=10) | 0.00253 ± 0.0012367 (n=10) | 2.0458 |
| FashionMNIST | hypersphere | 2000 | 250 | magnitude | 0.81454 ± 0.00076394 (n=10) | 0.00793 ± 0.0011887 (n=10) | 6.6714 |
| FashionMNIST | hypersphere | 2000 | 250 | obs | 0.83038 ± 0.00056388 (n=10) | 0.02377 ± 0.0010061 (n=10) | 23.626 |
| FashionMNIST | hypersphere | 2000 | 250 | obs_oneshot | 0.82318 ± 0.000748 (n=10) | 0.01657 ± 0.00091931 (n=10) | 18.024 |
| FashionMNIST | hypersphere | 4000 | 4000 | pool | 0.87364 ± 0.00056808 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | 4000 | 2000 | activity | 0.86119 ± 0.00034911 (n=10) | -0.00113 ± 0.0005459 (n=10) | -2.07 |
| FashionMNIST | hypersphere | 4000 | 2000 | magnitude | 0.86415 ± 0.00027376 (n=10) | 0.00183 ± 0.0004951 (n=10) | 3.6962 |
| FashionMNIST | hypersphere | 4000 | 2000 | obs | 0.86858 ± 0.00055614 (n=10) | 0.00626 ± 0.00043261 (n=10) | 14.47 |
| FashionMNIST | hypersphere | 4000 | 2000 | obs_oneshot | 0.86837 ± 0.00049127 (n=10) | 0.00605 ± 0.00068187 (n=10) | 8.8727 |
| FashionMNIST | hypersphere | 4000 | 1000 | activity | 0.84519 ± 0.00074154 (n=10) | -0.00364 ± 0.00077333 (n=10) | -4.7069 |
| FashionMNIST | hypersphere | 4000 | 1000 | magnitude | 0.84978 ± 0.00054829 (n=10) | 0.00095 ± 0.00057198 (n=10) | 1.6609 |
| FashionMNIST | hypersphere | 4000 | 1000 | obs | 0.86049 ± 0.00042647 (n=10) | 0.01166 ± 0.00054369 (n=10) | 21.446 |
| FashionMNIST | hypersphere | 4000 | 1000 | obs_oneshot | 0.85654 ± 0.00053897 (n=10) | 0.00771 ± 0.00054517 (n=10) | 14.142 |
| FashionMNIST | hypersphere | 4000 | 500 | activity | 0.82659 ± 0.00061 (n=10) | -0.00412 ± 0.00082338 (n=10) | -5.0038 |
| FashionMNIST | hypersphere | 4000 | 500 | magnitude | 0.83202 ± 0.00058344 (n=10) | 0.00131 ± 0.00068288 (n=10) | 1.9184 |
| FashionMNIST | hypersphere | 4000 | 500 | obs | 0.84835 ± 0.00062151 (n=10) | 0.01764 ± 0.001009 (n=10) | 17.483 |
| FashionMNIST | hypersphere | 4000 | 500 | obs_oneshot | 0.84046 ± 0.00068056 (n=10) | 0.00975 ± 0.00087041 (n=10) | 11.202 |
| FashionMNIST | hypersphere | 4000 | 250 | activity | 0.80696 ± 0.00073109 (n=10) | 0.00035 ± 0.0012184 (n=10) | 0.28726 |
| FashionMNIST | hypersphere | 4000 | 250 | magnitude | 0.81026 ± 0.00078996 (n=10) | 0.00365 ± 0.00069765 (n=10) | 5.2318 |
| FashionMNIST | hypersphere | 4000 | 250 | obs | 0.83337 ± 0.000709 (n=10) | 0.02676 ± 0.00083336 (n=10) | 32.111 |
| FashionMNIST | hypersphere | 4000 | 250 | obs_oneshot | 0.81983 ± 0.00082718 (n=10) | 0.01322 ± 0.0010525 (n=10) | 12.561 |
| FashionMNIST | hypersphere | 8000 | 8000 | pool | 0.88376 ± 0.00048401 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | hypersphere | 8000 | 2000 | activity | 0.85921 ± 0.00044832 (n=10) | -0.00311 ± 0.00087209 (n=10) | -3.5661 |
| FashionMNIST | hypersphere | 8000 | 2000 | magnitude | 0.86202 ± 0.00069663 (n=10) | -0.0003 ± 0.00078429 (n=10) | -0.38251 |
| FashionMNIST | hypersphere | 8000 | 2000 | obs | 0.87201 ± 0.00040012 (n=10) | 0.00969 ± 0.00058868 (n=10) | 16.461 |
| FashionMNIST | hypersphere | 8000 | 2000 | obs_oneshot | 0.86864 ± 0.00027536 (n=10) | 0.00632 ± 0.00076169 (n=10) | 8.2973 |
| FashionMNIST | hypersphere | 8000 | 1000 | activity | 0.84237 ± 0.0005224 (n=10) | -0.00646 ± 0.00042405 (n=10) | -15.234 |
| FashionMNIST | hypersphere | 8000 | 1000 | magnitude | 0.84513 ± 0.00058043 (n=10) | -0.0037 ± 0.00075366 (n=10) | -4.9094 |
| FashionMNIST | hypersphere | 8000 | 1000 | obs | 0.86297 ± 0.00044523 (n=10) | 0.01414 ± 0.00068673 (n=10) | 20.59 |
| FashionMNIST | hypersphere | 8000 | 1000 | obs_oneshot | 0.85691 ± 0.00055346 (n=10) | 0.00808 ± 0.00068261 (n=10) | 11.837 |
| FashionMNIST | hypersphere | 8000 | 500 | activity | 0.82551 ± 0.00052882 (n=10) | -0.0052 ± 0.00088066 (n=10) | -5.9047 |
| FashionMNIST | hypersphere | 8000 | 500 | magnitude | 0.82695 ± 0.00056159 (n=10) | -0.00376 ± 0.00083881 (n=10) | -4.4825 |
| FashionMNIST | hypersphere | 8000 | 500 | obs | 0.851 ± 0.00042714 (n=10) | 0.02029 ± 0.00056636 (n=10) | 35.825 |
| FashionMNIST | hypersphere | 8000 | 500 | obs_oneshot | 0.83997 ± 0.00089889 (n=10) | 0.00926 ± 0.0010582 (n=10) | 8.7506 |
| FashionMNIST | hypersphere | 8000 | 250 | activity | 0.80572 ± 0.00071473 (n=10) | -0.00089 ± 0.0013357 (n=10) | -0.66632 |
| FashionMNIST | hypersphere | 8000 | 250 | magnitude | 0.80706 ± 0.00059576 (n=10) | 0.00045 ± 0.0011638 (n=10) | 0.38665 |
| FashionMNIST | hypersphere | 8000 | 250 | obs | 0.83639 ± 0.00059263 (n=10) | 0.02978 ± 0.0010341 (n=10) | 28.799 |
| FashionMNIST | hypersphere | 8000 | 250 | obs_oneshot | 0.81643 ± 0.0011476 (n=10) | 0.00982 ± 0.0014112 (n=10) | 6.9586 |
| FashionMNIST | receptive_field | — | 250 | random | 0.80392 ± 0.00099239 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | — | 500 | random | 0.83007 ± 0.00074982 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | — | 750 | random | 0.84292 ± 0.00084456 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | — | 1000 | random | 0.85106 ± 0.0007032 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | — | 1500 | random | 0.86116 ± 0.00085922 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | — | 2000 | random | 0.86923 ± 0.00077331 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | — | 3000 | random | 0.87661 ± 0.00048476 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | — | 4000 | random | 0.88181 ± 0.00055707 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | — | 6000 | random | 0.8876 ± 0.00056569 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | — | 8000 | random | 0.89088 ± 0.00055594 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | 2000 | 2000 | pool | 0.86923 ± 0.00077331 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | 2000 | 1000 | activity | 0.84854 ± 0.00054041 (n=10) | -0.00252 ± 0.00061243 (n=10) | -4.1148 |
| FashionMNIST | receptive_field | 2000 | 1000 | magnitude | 0.85604 ± 0.00069685 (n=10) | 0.00498 ± 0.00084811 (n=10) | 5.8719 |
| FashionMNIST | receptive_field | 2000 | 1000 | obs | 0.86285 ± 0.00068496 (n=10) | 0.01179 ± 0.00080587 (n=10) | 14.63 |
| FashionMNIST | receptive_field | 2000 | 1000 | obs_oneshot | 0.86202 ± 0.00061006 (n=10) | 0.01096 ± 0.00076583 (n=10) | 14.311 |
| FashionMNIST | receptive_field | 2000 | 500 | activity | 0.82564 ± 0.00054041 (n=10) | -0.00443 ± 0.00058081 (n=10) | -7.6272 |
| FashionMNIST | receptive_field | 2000 | 500 | magnitude | 0.83485 ± 0.0006745 (n=10) | 0.00478 ± 0.0010931 (n=10) | 4.3729 |
| FashionMNIST | receptive_field | 2000 | 500 | obs | 0.85108 ± 0.00081946 (n=10) | 0.02101 ± 0.0013534 (n=10) | 15.524 |
| FashionMNIST | receptive_field | 2000 | 500 | obs_oneshot | 0.84913 ± 0.00069251 (n=10) | 0.01906 ± 0.0013118 (n=10) | 14.53 |
| FashionMNIST | receptive_field | 2000 | 250 | activity | 0.80293 ± 0.0014372 (n=10) | -0.00099 ± 0.0018669 (n=10) | -0.53028 |
| FashionMNIST | receptive_field | 2000 | 250 | magnitude | 0.81415 ± 0.0010208 (n=10) | 0.01023 ± 0.0015129 (n=10) | 6.7618 |
| FashionMNIST | receptive_field | 2000 | 250 | obs | 0.83562 ± 0.00089067 (n=10) | 0.0317 ± 0.0014919 (n=10) | 21.248 |
| FashionMNIST | receptive_field | 2000 | 250 | obs_oneshot | 0.82752 ± 0.0005489 (n=10) | 0.0236 ± 0.0010477 (n=10) | 22.524 |
| FashionMNIST | receptive_field | 4000 | 4000 | pool | 0.88181 ± 0.00055707 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | 4000 | 2000 | activity | 0.86558 ± 0.00060033 (n=10) | -0.00365 ± 0.00059521 (n=10) | -6.1323 |
| FashionMNIST | receptive_field | 4000 | 2000 | magnitude | 0.87072 ± 0.00072844 (n=10) | 0.00149 ± 0.00076644 (n=10) | 1.944 |
| FashionMNIST | receptive_field | 4000 | 2000 | obs | 0.87719 ± 0.0006023 (n=10) | 0.00796 ± 0.00070604 (n=10) | 11.274 |
| FashionMNIST | receptive_field | 4000 | 2000 | obs_oneshot | 0.87605 ± 0.00055543 (n=10) | 0.00682 ± 0.00074383 (n=10) | 9.1687 |
| FashionMNIST | receptive_field | 4000 | 1000 | activity | 0.84509 ± 0.00047667 (n=10) | -0.00597 ± 0.0007975 (n=10) | -7.4859 |
| FashionMNIST | receptive_field | 4000 | 1000 | magnitude | 0.853 ± 0.00052345 (n=10) | 0.00194 ± 0.00061431 (n=10) | 3.158 |
| FashionMNIST | receptive_field | 4000 | 1000 | obs | 0.86917 ± 0.0007513 (n=10) | 0.01811 ± 0.0011521 (n=10) | 15.719 |
| FashionMNIST | receptive_field | 4000 | 1000 | obs_oneshot | 0.86621 ± 0.00057493 (n=10) | 0.01515 ± 0.00067169 (n=10) | 22.555 |
| FashionMNIST | receptive_field | 4000 | 500 | activity | 0.82477 ± 0.00073167 (n=10) | -0.0053 ± 0.00085258 (n=10) | -6.2164 |
| FashionMNIST | receptive_field | 4000 | 500 | magnitude | 0.83302 ± 0.00078483 (n=10) | 0.00295 ± 0.0012045 (n=10) | 2.4492 |
| FashionMNIST | receptive_field | 4000 | 500 | obs | 0.85733 ± 0.00062415 (n=10) | 0.02726 ± 0.0011723 (n=10) | 23.254 |
| FashionMNIST | receptive_field | 4000 | 500 | obs_oneshot | 0.84991 ± 0.00068742 (n=10) | 0.01984 ± 0.0010608 (n=10) | 18.702 |
| FashionMNIST | receptive_field | 4000 | 250 | activity | 0.80442 ± 0.0010012 (n=10) | 0.0005 ± 0.0015345 (n=10) | 0.32584 |
| FashionMNIST | receptive_field | 4000 | 250 | magnitude | 0.81268 ± 0.00095135 (n=10) | 0.00876 ± 0.0014084 (n=10) | 6.2198 |
| FashionMNIST | receptive_field | 4000 | 250 | obs | 0.84136 ± 0.00076568 (n=10) | 0.03744 ± 0.0014666 (n=10) | 25.528 |
| FashionMNIST | receptive_field | 4000 | 250 | obs_oneshot | 0.8246 ± 0.00084406 (n=10) | 0.02068 ± 0.0011689 (n=10) | 17.691 |
| FashionMNIST | receptive_field | 8000 | 8000 | pool | 0.89088 ± 0.00055594 (n=10) | 0 ± 0 (n=10) | 0 |
| FashionMNIST | receptive_field | 8000 | 2000 | activity | 0.86191 ± 0.00067913 (n=10) | -0.00732 ± 0.00086664 (n=10) | -8.4464 |
| FashionMNIST | receptive_field | 8000 | 2000 | magnitude | 0.86802 ± 0.00056131 (n=10) | -0.00121 ± 0.0007808 (n=10) | -1.5497 |
| FashionMNIST | receptive_field | 8000 | 2000 | obs | 0.88158 ± 0.00042158 (n=10) | 0.01235 ± 0.00087041 (n=10) | 14.189 |
| FashionMNIST | receptive_field | 8000 | 2000 | obs_oneshot | 0.87945 ± 0.00055161 (n=10) | 0.01022 ± 0.0010775 (n=10) | 9.4847 |
| FashionMNIST | receptive_field | 8000 | 1000 | activity | 0.84376 ± 0.00055682 (n=10) | -0.0073 ± 0.0007723 (n=10) | -9.4523 |
| FashionMNIST | receptive_field | 8000 | 1000 | magnitude | 0.84945 ± 0.00058959 (n=10) | -0.00161 ± 0.0010072 (n=10) | -1.5984 |
| FashionMNIST | receptive_field | 8000 | 1000 | obs | 0.87182 ± 0.00051441 (n=10) | 0.02076 ± 0.00097504 (n=10) | 21.291 |
| FashionMNIST | receptive_field | 8000 | 1000 | obs_oneshot | 0.86627 ± 0.00059237 (n=10) | 0.01521 ± 0.0011155 (n=10) | 13.635 |
| FashionMNIST | receptive_field | 8000 | 500 | activity | 0.82414 ± 0.00067958 (n=10) | -0.00593 ± 0.00086885 (n=10) | -6.8251 |
| FashionMNIST | receptive_field | 8000 | 500 | magnitude | 0.8302 ± 0.0008184 (n=10) | 0.00013 ± 0.0012657 (n=10) | 0.10271 |
| FashionMNIST | receptive_field | 8000 | 500 | obs | 0.86095 ± 0.00042668 (n=10) | 0.03088 ± 0.00091722 (n=10) | 33.667 |
| FashionMNIST | receptive_field | 8000 | 500 | obs_oneshot | 0.8467 ± 0.00081677 (n=10) | 0.01663 ± 0.0011685 (n=10) | 14.232 |
| FashionMNIST | receptive_field | 8000 | 250 | activity | 0.80385 ± 0.00091217 (n=10) | -7e-05 ± 0.0011729 (n=10) | -0.059679 |
| FashionMNIST | receptive_field | 8000 | 250 | magnitude | 0.80719 ± 0.00094874 (n=10) | 0.00327 ± 0.0014352 (n=10) | 2.2784 |
| FashionMNIST | receptive_field | 8000 | 250 | obs | 0.84419 ± 0.00058546 (n=10) | 0.04027 ± 0.0012216 (n=10) | 32.966 |
| FashionMNIST | receptive_field | 8000 | 250 | obs_oneshot | 0.8195 ± 0.001091 (n=10) | 0.01558 ± 0.0014263 (n=10) | 10.923 |
| CIFAR10 | hypersphere | — | 250 | random | 0.40147 ± 0.0011371 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | — | 500 | random | 0.43228 ± 0.0010183 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | — | 750 | random | 0.44598 ± 0.0010676 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | — | 1000 | random | 0.4559 ± 0.001057 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | — | 1500 | random | 0.46876 ± 0.001183 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | — | 2000 | random | 0.47757 ± 0.00075027 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | — | 3000 | random | 0.48719 ± 0.00091341 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | — | 4000 | random | 0.49233 ± 0.00083799 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | — | 6000 | random | 0.49977 ± 0.00070491 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | — | 8000 | random | 0.50312 ± 0.00098847 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | 2000 | 2000 | pool | 0.47757 ± 0.00075027 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | 2000 | 1000 | activity | 0.45382 ± 0.00089923 (n=10) | -0.00208 ± 0.001275 (n=10) | -1.6313 |
| CIFAR10 | hypersphere | 2000 | 1000 | magnitude | 0.45776 ± 0.0009413 (n=10) | 0.00186 ± 0.00095303 (n=10) | 1.9517 |
| CIFAR10 | hypersphere | 2000 | 1000 | obs | 0.46653 ± 0.00082436 (n=10) | 0.01063 ± 0.0013124 (n=10) | 8.0995 |
| CIFAR10 | hypersphere | 2000 | 1000 | obs_oneshot | 0.46483 ± 0.00079736 (n=10) | 0.00893 ± 0.0010118 (n=10) | 8.8256 |
| CIFAR10 | hypersphere | 2000 | 500 | activity | 0.42809 ± 0.0010252 (n=10) | -0.00419 ± 0.0018352 (n=10) | -2.2831 |
| CIFAR10 | hypersphere | 2000 | 500 | magnitude | 0.42872 ± 0.0010173 (n=10) | -0.00356 ± 0.0011851 (n=10) | -3.0039 |
| CIFAR10 | hypersphere | 2000 | 500 | obs | 0.45006 ± 0.0011407 (n=10) | 0.01778 ± 0.001407 (n=10) | 12.637 |
| CIFAR10 | hypersphere | 2000 | 500 | obs_oneshot | 0.44237 ± 0.00097525 (n=10) | 0.01009 ± 0.0013608 (n=10) | 7.4146 |
| CIFAR10 | hypersphere | 2000 | 250 | activity | 0.39792 ± 0.0011255 (n=10) | -0.00355 ± 0.0016294 (n=10) | -2.1787 |
| CIFAR10 | hypersphere | 2000 | 250 | magnitude | 0.39991 ± 0.0008915 (n=10) | -0.00156 ± 0.0016742 (n=10) | -0.93179 |
| CIFAR10 | hypersphere | 2000 | 250 | obs | 0.43117 ± 0.00095998 (n=10) | 0.0297 ± 0.0011467 (n=10) | 25.901 |
| CIFAR10 | hypersphere | 2000 | 250 | obs_oneshot | 0.4159 ± 0.0013177 (n=10) | 0.01443 ± 0.0017476 (n=10) | 8.2569 |
| CIFAR10 | hypersphere | 4000 | 4000 | pool | 0.49233 ± 0.00083799 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | 4000 | 2000 | activity | 0.47294 ± 0.0012577 (n=10) | -0.00463 ± 0.0015333 (n=10) | -3.0197 |
| CIFAR10 | hypersphere | 4000 | 2000 | magnitude | 0.47388 ± 0.0010735 (n=10) | -0.00369 ± 0.0012529 (n=10) | -2.9453 |
| CIFAR10 | hypersphere | 4000 | 2000 | obs | 0.48467 ± 0.0009579 (n=10) | 0.0071 ± 0.0012232 (n=10) | 5.8044 |
| CIFAR10 | hypersphere | 4000 | 2000 | obs_oneshot | 0.48284 ± 0.00068964 (n=10) | 0.00527 ± 0.001008 (n=10) | 5.2283 |
| CIFAR10 | hypersphere | 4000 | 1000 | activity | 0.44766 ± 0.0011883 (n=10) | -0.00824 ± 0.0016591 (n=10) | -4.9667 |
| CIFAR10 | hypersphere | 4000 | 1000 | magnitude | 0.44719 ± 0.0011209 (n=10) | -0.00871 ± 0.0010777 (n=10) | -8.082 |
| CIFAR10 | hypersphere | 4000 | 1000 | obs | 0.4707 ± 0.0007912 (n=10) | 0.0148 ± 0.0015304 (n=10) | 9.6709 |
| CIFAR10 | hypersphere | 4000 | 1000 | obs_oneshot | 0.46336 ± 0.0010559 (n=10) | 0.00746 ± 0.0016977 (n=10) | 4.3941 |
| CIFAR10 | hypersphere | 4000 | 500 | activity | 0.42119 ± 0.00072349 (n=10) | -0.01109 ± 0.0012839 (n=10) | -8.6379 |
| CIFAR10 | hypersphere | 4000 | 500 | magnitude | 0.42024 ± 0.0013689 (n=10) | -0.01204 ± 0.0016206 (n=10) | -7.4291 |
| CIFAR10 | hypersphere | 4000 | 500 | obs | 0.45471 ± 0.0010049 (n=10) | 0.02243 ± 0.0016581 (n=10) | 13.527 |
| CIFAR10 | hypersphere | 4000 | 500 | obs_oneshot | 0.43984 ± 0.0005818 (n=10) | 0.00756 ± 0.0011582 (n=10) | 6.5275 |
| CIFAR10 | hypersphere | 4000 | 250 | activity | 0.39434 ± 0.00096393 (n=10) | -0.00713 ± 0.0016301 (n=10) | -4.3739 |
| CIFAR10 | hypersphere | 4000 | 250 | magnitude | 0.39119 ± 0.0014237 (n=10) | -0.01028 ± 0.0022144 (n=10) | -4.6424 |
| CIFAR10 | hypersphere | 4000 | 250 | obs | 0.43413 ± 0.00065473 (n=10) | 0.03266 ± 0.0013359 (n=10) | 24.447 |
| CIFAR10 | hypersphere | 4000 | 250 | obs_oneshot | 0.41217 ± 0.0010116 (n=10) | 0.0107 ± 0.0010042 (n=10) | 10.655 |
| CIFAR10 | hypersphere | 8000 | 8000 | pool | 0.50312 ± 0.00098847 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | hypersphere | 8000 | 2000 | activity | 0.46704 ± 0.0008631 (n=10) | -0.01053 ± 0.001305 (n=10) | -8.0693 |
| CIFAR10 | hypersphere | 8000 | 2000 | magnitude | 0.4641 ± 0.0010196 (n=10) | -0.01347 ± 0.0012958 (n=10) | -10.395 |
| CIFAR10 | hypersphere | 8000 | 2000 | obs | 0.48571 ± 0.0011966 (n=10) | 0.00814 ± 0.00072544 (n=10) | 11.221 |
| CIFAR10 | hypersphere | 8000 | 2000 | obs_oneshot | 0.48107 ± 0.00081214 (n=10) | 0.0035 ± 0.00093405 (n=10) | 3.7471 |
| CIFAR10 | hypersphere | 8000 | 1000 | activity | 0.44136 ± 0.0011774 (n=10) | -0.01454 ± 0.0013929 (n=10) | -10.439 |
| CIFAR10 | hypersphere | 8000 | 1000 | magnitude | 0.43698 ± 0.00098102 (n=10) | -0.01892 ± 0.0010863 (n=10) | -17.418 |
| CIFAR10 | hypersphere | 8000 | 1000 | obs | 0.47275 ± 0.0012689 (n=10) | 0.01685 ± 0.0019946 (n=10) | 8.4477 |
| CIFAR10 | hypersphere | 8000 | 1000 | obs_oneshot | 0.4608 ± 0.00089007 (n=10) | 0.0049 ± 0.0015117 (n=10) | 3.2415 |
| CIFAR10 | hypersphere | 8000 | 500 | activity | 0.41676 ± 0.00054878 (n=10) | -0.01552 ± 0.0011967 (n=10) | -12.969 |
| CIFAR10 | hypersphere | 8000 | 500 | magnitude | 0.40925 ± 0.00070147 (n=10) | -0.02303 ± 0.0014228 (n=10) | -16.186 |
| CIFAR10 | hypersphere | 8000 | 500 | obs | 0.45624 ± 0.00091848 (n=10) | 0.02396 ± 0.00151 (n=10) | 15.868 |
| CIFAR10 | hypersphere | 8000 | 500 | obs_oneshot | 0.43617 ± 0.00097126 (n=10) | 0.00389 ± 0.00094156 (n=10) | 4.1314 |
| CIFAR10 | hypersphere | 8000 | 250 | activity | 0.39162 ± 0.00092482 (n=10) | -0.00985 ± 0.0013604 (n=10) | -7.2405 |
| CIFAR10 | hypersphere | 8000 | 250 | magnitude | 0.38739 ± 0.00085433 (n=10) | -0.01408 ± 0.0011932 (n=10) | -11.8 |
| CIFAR10 | hypersphere | 8000 | 250 | obs | 0.43385 ± 0.00066203 (n=10) | 0.03238 ± 0.001103 (n=10) | 29.356 |
| CIFAR10 | hypersphere | 8000 | 250 | obs_oneshot | 0.40408 ± 0.0014245 (n=10) | 0.00261 ± 0.0013632 (n=10) | 1.9146 |
| CIFAR10 | receptive_field | — | 250 | random | 0.41831 ± 0.0011677 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | — | 500 | random | 0.45501 ± 0.0015386 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | — | 750 | random | 0.4772 ± 0.0015088 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | — | 1000 | random | 0.48941 ± 0.0014821 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | — | 1500 | random | 0.50723 ± 0.0012061 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | — | 2000 | random | 0.51845 ± 0.0012215 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | — | 3000 | random | 0.53635 ± 0.00073971 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | — | 4000 | random | 0.54558 ± 0.00072553 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | — | 6000 | random | 0.55641 ± 0.0006819 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | — | 8000 | random | 0.5618 ± 0.00073258 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | 2000 | 2000 | pool | 0.51845 ± 0.0012215 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | 2000 | 1000 | activity | 0.4756 ± 0.00073045 (n=10) | -0.01381 ± 0.0014955 (n=10) | -9.2343 |
| CIFAR10 | receptive_field | 2000 | 1000 | magnitude | 0.4831 ± 0.0013077 (n=10) | -0.00631 ± 0.0016471 (n=10) | -3.8309 |
| CIFAR10 | receptive_field | 2000 | 1000 | obs | 0.50959 ± 0.00082952 (n=10) | 0.02018 ± 0.0014914 (n=10) | 13.531 |
| CIFAR10 | receptive_field | 2000 | 1000 | obs_oneshot | 0.50751 ± 0.00086042 (n=10) | 0.0181 ± 0.0014575 (n=10) | 12.419 |
| CIFAR10 | receptive_field | 2000 | 500 | activity | 0.43248 ± 0.0012216 (n=10) | -0.02253 ± 0.0012663 (n=10) | -17.792 |
| CIFAR10 | receptive_field | 2000 | 500 | magnitude | 0.43548 ± 0.0017578 (n=10) | -0.01953 ± 0.0023329 (n=10) | -8.3717 |
| CIFAR10 | receptive_field | 2000 | 500 | obs | 0.49389 ± 0.00083046 (n=10) | 0.03888 ± 0.0015778 (n=10) | 24.642 |
| CIFAR10 | receptive_field | 2000 | 500 | obs_oneshot | 0.4865 ± 0.0012728 (n=10) | 0.03149 ± 0.0014459 (n=10) | 21.778 |
| CIFAR10 | receptive_field | 2000 | 250 | activity | 0.39389 ± 0.001006 (n=10) | -0.02442 ± 0.00083544 (n=10) | -29.23 |
| CIFAR10 | receptive_field | 2000 | 250 | magnitude | 0.40055 ± 0.0015318 (n=10) | -0.01776 ± 0.0013673 (n=10) | -12.989 |
| CIFAR10 | receptive_field | 2000 | 250 | obs | 0.4692 ± 0.0016432 (n=10) | 0.05089 ± 0.0022153 (n=10) | 22.972 |
| CIFAR10 | receptive_field | 2000 | 250 | obs_oneshot | 0.45419 ± 0.0014764 (n=10) | 0.03588 ± 0.0016738 (n=10) | 21.436 |
| CIFAR10 | receptive_field | 4000 | 4000 | pool | 0.54558 ± 0.00072553 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | 4000 | 2000 | activity | 0.50314 ± 0.0012571 (n=10) | -0.01531 ± 0.0018205 (n=10) | -8.4099 |
| CIFAR10 | receptive_field | 4000 | 2000 | magnitude | 0.50525 ± 0.0010925 (n=10) | -0.0132 ± 0.0013573 (n=10) | -9.7253 |
| CIFAR10 | receptive_field | 4000 | 2000 | obs | 0.53737 ± 0.0012603 (n=10) | 0.01892 ± 0.0015941 (n=10) | 11.868 |
| CIFAR10 | receptive_field | 4000 | 2000 | obs_oneshot | 0.53406 ± 0.00083854 (n=10) | 0.01561 ± 0.001757 (n=10) | 8.8845 |
| CIFAR10 | receptive_field | 4000 | 1000 | activity | 0.46044 ± 0.0010149 (n=10) | -0.02897 ± 0.0016945 (n=10) | -17.096 |
| CIFAR10 | receptive_field | 4000 | 1000 | magnitude | 0.45965 ± 0.00094319 (n=10) | -0.02976 ± 0.0015971 (n=10) | -18.634 |
| CIFAR10 | receptive_field | 4000 | 1000 | obs | 0.52063 ± 0.00055218 (n=10) | 0.03122 ± 0.0016997 (n=10) | 18.368 |
| CIFAR10 | receptive_field | 4000 | 1000 | obs_oneshot | 0.51407 ± 0.00090481 (n=10) | 0.02466 ± 0.0015884 (n=10) | 15.525 |
| CIFAR10 | receptive_field | 4000 | 500 | activity | 0.42235 ± 0.00088396 (n=10) | -0.03266 ± 0.0013835 (n=10) | -23.607 |
| CIFAR10 | receptive_field | 4000 | 500 | magnitude | 0.42083 ± 0.00099711 (n=10) | -0.03418 ± 0.0018739 (n=10) | -18.24 |
| CIFAR10 | receptive_field | 4000 | 500 | obs | 0.50123 ± 0.00068848 (n=10) | 0.04622 ± 0.0013661 (n=10) | 33.834 |
| CIFAR10 | receptive_field | 4000 | 500 | obs_oneshot | 0.48643 ± 0.00071834 (n=10) | 0.03142 ± 0.0016032 (n=10) | 19.598 |
| CIFAR10 | receptive_field | 4000 | 250 | activity | 0.39388 ± 0.0010595 (n=10) | -0.02443 ± 0.0017139 (n=10) | -14.254 |
| CIFAR10 | receptive_field | 4000 | 250 | magnitude | 0.3921 ± 0.0013296 (n=10) | -0.02621 ± 0.0013918 (n=10) | -18.832 |
| CIFAR10 | receptive_field | 4000 | 250 | obs | 0.47501 ± 0.00087426 (n=10) | 0.0567 ± 0.001353 (n=10) | 41.906 |
| CIFAR10 | receptive_field | 4000 | 250 | obs_oneshot | 0.45132 ± 0.0023679 (n=10) | 0.03301 ± 0.0029017 (n=10) | 11.376 |
| CIFAR10 | receptive_field | 8000 | 8000 | pool | 0.5618 ± 0.00073258 (n=10) | 0 ± 0 (n=10) | 0 |
| CIFAR10 | receptive_field | 8000 | 2000 | activity | 0.4858 ± 0.0012221 (n=10) | -0.03265 ± 0.0020729 (n=10) | -15.751 |
| CIFAR10 | receptive_field | 8000 | 2000 | magnitude | 0.4787 ± 0.00087242 (n=10) | -0.03975 ± 0.0015685 (n=10) | -25.343 |
| CIFAR10 | receptive_field | 8000 | 2000 | obs | 0.54538 ± 0.00080509 (n=10) | 0.02693 ± 0.0012953 (n=10) | 20.791 |
| CIFAR10 | receptive_field | 8000 | 2000 | obs_oneshot | 0.537 ± 0.00083905 (n=10) | 0.01855 ± 0.0016164 (n=10) | 11.476 |
| CIFAR10 | receptive_field | 8000 | 1000 | activity | 0.44847 ± 0.0012769 (n=10) | -0.04094 ± 0.0017202 (n=10) | -23.799 |
| CIFAR10 | receptive_field | 8000 | 1000 | magnitude | 0.44051 ± 0.0014198 (n=10) | -0.0489 ± 0.0015102 (n=10) | -32.38 |
| CIFAR10 | receptive_field | 8000 | 1000 | obs | 0.52714 ± 0.00080197 (n=10) | 0.03773 ± 0.0016762 (n=10) | 22.509 |
| CIFAR10 | receptive_field | 8000 | 1000 | obs_oneshot | 0.51217 ± 0.00098184 (n=10) | 0.02276 ± 0.001736 (n=10) | 13.11 |
| CIFAR10 | receptive_field | 8000 | 500 | activity | 0.41508 ± 0.00081251 (n=10) | -0.03993 ± 0.001637 (n=10) | -24.392 |
| CIFAR10 | receptive_field | 8000 | 500 | magnitude | 0.4108 ± 0.0017614 (n=10) | -0.04421 ± 0.0024981 (n=10) | -17.698 |
| CIFAR10 | receptive_field | 8000 | 500 | obs | 0.50852 ± 0.0010563 (n=10) | 0.05351 ± 0.0019893 (n=10) | 26.899 |
| CIFAR10 | receptive_field | 8000 | 500 | obs_oneshot | 0.48155 ± 0.0013039 (n=10) | 0.02654 ± 0.0025595 (n=10) | 10.369 |
| CIFAR10 | receptive_field | 8000 | 250 | activity | 0.39174 ± 0.0011289 (n=10) | -0.02657 ± 0.001765 (n=10) | -15.054 |
| CIFAR10 | receptive_field | 8000 | 250 | magnitude | 0.38888 ± 0.00099071 (n=10) | -0.02943 ± 0.0015067 (n=10) | -19.533 |
| CIFAR10 | receptive_field | 8000 | 250 | obs | 0.48038 ± 0.00080578 (n=10) | 0.06207 ± 0.0017414 (n=10) | 35.644 |
| CIFAR10 | receptive_field | 8000 | 250 | obs_oneshot | 0.44342 ± 0.0012 (n=10) | 0.02511 ± 0.0015403 (n=10) | 16.302 |
| MNIST (exploratory) | hypersphere | — | 250 | random | 0.89043 ± 0.00078642 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | — | 500 | random | 0.92338 ± 0.00086715 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | — | 750 | random | 0.93652 ± 0.00048438 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | — | 1000 | random | 0.94414 ± 0.00048056 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | — | 1500 | random | 0.95286 ± 0.00049603 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | — | 2000 | random | 0.95866 ± 0.00036914 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | — | 3000 | random | 0.96398 ± 0.0003881 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | — | 4000 | random | 0.96816 ± 0.00032769 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | — | 6000 | random | 0.97164 ± 0.00023485 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | — | 8000 | random | 0.97365 ± 0.00021409 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | 2000 | 2000 | pool | 0.95866 ± 0.00036914 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | 2000 | 1000 | activity | 0.94249 ± 0.0005227 (n=10) | -0.00165 ± 0.00064192 (n=10) | -2.5704 |
| MNIST (exploratory) | hypersphere | 2000 | 1000 | magnitude | 0.94724 ± 0.00043874 (n=10) | 0.0031 ± 0.00027809 (n=10) | 11.148 |
| MNIST (exploratory) | hypersphere | 2000 | 1000 | obs | 0.95155 ± 0.00049872 (n=10) | 0.00741 ± 0.00068919 (n=10) | 10.752 |
| MNIST (exploratory) | hypersphere | 2000 | 1000 | obs_oneshot | 0.95146 ± 0.00042693 (n=10) | 0.00732 ± 0.00058115 (n=10) | 12.596 |
| MNIST (exploratory) | hypersphere | 2000 | 500 | activity | 0.91708 ± 0.00076432 (n=10) | -0.0063 ± 0.001166 (n=10) | -5.4031 |
| MNIST (exploratory) | hypersphere | 2000 | 500 | magnitude | 0.9236 ± 0.00049733 (n=10) | 0.00022 ± 0.00099898 (n=10) | 0.22023 |
| MNIST (exploratory) | hypersphere | 2000 | 500 | obs | 0.94097 ± 0.00065099 (n=10) | 0.01759 ± 0.0011343 (n=10) | 15.508 |
| MNIST (exploratory) | hypersphere | 2000 | 500 | obs_oneshot | 0.93721 ± 0.00060927 (n=10) | 0.01383 ± 0.00099889 (n=10) | 13.845 |
| MNIST (exploratory) | hypersphere | 2000 | 250 | activity | 0.88422 ± 0.0010508 (n=10) | -0.00621 ± 0.0012016 (n=10) | -5.168 |
| MNIST (exploratory) | hypersphere | 2000 | 250 | magnitude | 0.88939 ± 0.00051971 (n=10) | -0.00104 ± 0.0010863 (n=10) | -0.95738 |
| MNIST (exploratory) | hypersphere | 2000 | 250 | obs | 0.92288 ± 0.00069774 (n=10) | 0.03245 ± 0.0011226 (n=10) | 28.906 |
| MNIST (exploratory) | hypersphere | 2000 | 250 | obs_oneshot | 0.91079 ± 0.00074884 (n=10) | 0.02036 ± 0.00088182 (n=10) | 23.089 |
| MNIST (exploratory) | hypersphere | 4000 | 4000 | pool | 0.96816 ± 0.00032769 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | 4000 | 2000 | activity | 0.95709 ± 0.0005001 (n=10) | -0.00157 ± 0.00045948 (n=10) | -3.4169 |
| MNIST (exploratory) | hypersphere | 4000 | 2000 | magnitude | 0.96018 ± 0.00046828 (n=10) | 0.00152 ± 0.00058325 (n=10) | 2.6061 |
| MNIST (exploratory) | hypersphere | 4000 | 2000 | obs | 0.96394 ± 0.00031559 (n=10) | 0.00528 ± 0.0003881 (n=10) | 13.605 |
| MNIST (exploratory) | hypersphere | 4000 | 2000 | obs_oneshot | 0.96355 ± 0.00028018 (n=10) | 0.00489 ± 0.00041911 (n=10) | 11.668 |
| MNIST (exploratory) | hypersphere | 4000 | 1000 | activity | 0.93772 ± 0.00061785 (n=10) | -0.00642 ± 0.00064787 (n=10) | -9.9094 |
| MNIST (exploratory) | hypersphere | 4000 | 1000 | magnitude | 0.94153 ± 0.0006515 (n=10) | -0.00261 ± 0.0007218 (n=10) | -3.616 |
| MNIST (exploratory) | hypersphere | 4000 | 1000 | obs | 0.95641 ± 0.00032368 (n=10) | 0.01227 ± 0.00049756 (n=10) | 24.66 |
| MNIST (exploratory) | hypersphere | 4000 | 1000 | obs_oneshot | 0.95486 ± 0.00045758 (n=10) | 0.01072 ± 0.00071318 (n=10) | 15.031 |
| MNIST (exploratory) | hypersphere | 4000 | 500 | activity | 0.91004 ± 0.00080763 (n=10) | -0.01334 ± 0.0011601 (n=10) | -11.499 |
| MNIST (exploratory) | hypersphere | 4000 | 500 | magnitude | 0.90982 ± 0.00093889 (n=10) | -0.01356 ± 0.0012975 (n=10) | -10.451 |
| MNIST (exploratory) | hypersphere | 4000 | 500 | obs | 0.9458 ± 0.00041526 (n=10) | 0.02242 ± 0.0010766 (n=10) | 20.825 |
| MNIST (exploratory) | hypersphere | 4000 | 500 | obs_oneshot | 0.93933 ± 0.00055758 (n=10) | 0.01595 ± 0.0012048 (n=10) | 13.238 |
| MNIST (exploratory) | hypersphere | 4000 | 250 | activity | 0.87822 ± 0.00098689 (n=10) | -0.01221 ± 0.0014453 (n=10) | -8.4483 |
| MNIST (exploratory) | hypersphere | 4000 | 250 | magnitude | 0.87931 ± 0.00079084 (n=10) | -0.01112 ± 0.0011926 (n=10) | -9.3238 |
| MNIST (exploratory) | hypersphere | 4000 | 250 | obs | 0.92893 ± 0.00056609 (n=10) | 0.0385 ± 0.0012406 (n=10) | 31.033 |
| MNIST (exploratory) | hypersphere | 4000 | 250 | obs_oneshot | 0.91495 ± 0.0010022 (n=10) | 0.02452 ± 0.0013727 (n=10) | 17.862 |
| MNIST (exploratory) | hypersphere | 8000 | 8000 | pool | 0.97365 ± 0.00021409 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | hypersphere | 8000 | 2000 | activity | 0.95498 ± 0.00040683 (n=10) | -0.00368 ± 0.00030869 (n=10) | -11.921 |
| MNIST (exploratory) | hypersphere | 8000 | 2000 | magnitude | 0.95512 ± 0.00023795 (n=10) | -0.00354 ± 0.00046457 (n=10) | -7.62 |
| MNIST (exploratory) | hypersphere | 8000 | 2000 | obs | 0.9666 ± 0.00038035 (n=10) | 0.00794 ± 0.00057777 (n=10) | 13.742 |
| MNIST (exploratory) | hypersphere | 8000 | 2000 | obs_oneshot | 0.96494 ± 0.00039531 (n=10) | 0.00628 ± 0.00058134 (n=10) | 10.803 |
| MNIST (exploratory) | hypersphere | 8000 | 1000 | activity | 0.93107 ± 0.0006443 (n=10) | -0.01307 ± 0.00066166 (n=10) | -19.753 |
| MNIST (exploratory) | hypersphere | 8000 | 1000 | magnitude | 0.93025 ± 0.0006745 (n=10) | -0.01389 ± 0.00071654 (n=10) | -19.385 |
| MNIST (exploratory) | hypersphere | 8000 | 1000 | obs | 0.95938 ± 0.00040628 (n=10) | 0.01524 ± 0.0006353 (n=10) | 23.989 |
| MNIST (exploratory) | hypersphere | 8000 | 1000 | obs_oneshot | 0.9546 ± 0.00024766 (n=10) | 0.01046 ± 0.00061774 (n=10) | 16.933 |
| MNIST (exploratory) | hypersphere | 8000 | 500 | activity | 0.8992 ± 0.00076522 (n=10) | -0.02418 ± 0.0013667 (n=10) | -17.692 |
| MNIST (exploratory) | hypersphere | 8000 | 500 | magnitude | 0.90014 ± 0.00073352 (n=10) | -0.02324 ± 0.0012874 (n=10) | -18.052 |
| MNIST (exploratory) | hypersphere | 8000 | 500 | obs | 0.94851 ± 0.00056243 (n=10) | 0.02513 ± 0.00075646 (n=10) | 33.22 |
| MNIST (exploratory) | hypersphere | 8000 | 500 | obs_oneshot | 0.9389 ± 0.00035434 (n=10) | 0.01552 ± 0.000951 (n=10) | 16.32 |
| MNIST (exploratory) | hypersphere | 8000 | 250 | activity | 0.87284 ± 0.00089495 (n=10) | -0.01759 ± 0.00099414 (n=10) | -17.694 |
| MNIST (exploratory) | hypersphere | 8000 | 250 | magnitude | 0.87447 ± 0.00095161 (n=10) | -0.01596 ± 0.0015034 (n=10) | -10.616 |
| MNIST (exploratory) | hypersphere | 8000 | 250 | obs | 0.93155 ± 0.00053443 (n=10) | 0.04112 ± 0.00063523 (n=10) | 64.733 |
| MNIST (exploratory) | hypersphere | 8000 | 250 | obs_oneshot | 0.91216 ± 0.00091642 (n=10) | 0.02173 ± 0.0010893 (n=10) | 19.948 |
| MNIST (exploratory) | receptive_field | — | 250 | random | 0.88866 ± 0.0025674 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | — | 500 | random | 0.92382 ± 0.0012736 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | — | 750 | random | 0.93981 ± 0.00075151 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | — | 1000 | random | 0.94772 ± 0.00052021 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | — | 1500 | random | 0.95754 ± 0.0002242 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | — | 2000 | random | 0.96278 ± 0.00036295 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | — | 3000 | random | 0.96823 ± 0.00027246 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | — | 4000 | random | 0.97092 ± 0.00043071 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | — | 6000 | random | 0.97428 ± 0.00029242 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | — | 8000 | random | 0.97651 ± 0.00024696 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | 2000 | 2000 | pool | 0.96278 ± 0.00036295 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | 2000 | 1000 | activity | 0.95747 ± 0.0003896 (n=10) | 0.00975 ± 0.00068203 (n=10) | 14.296 |
| MNIST (exploratory) | receptive_field | 2000 | 1000 | magnitude | 0.95956 ± 0.00045294 (n=10) | 0.01184 ± 0.00070225 (n=10) | 16.86 |
| MNIST (exploratory) | receptive_field | 2000 | 1000 | obs | 0.96064 ± 0.00037423 (n=10) | 0.01292 ± 0.00063802 (n=10) | 20.25 |
| MNIST (exploratory) | receptive_field | 2000 | 1000 | obs_oneshot | 0.96094 ± 0.00032359 (n=10) | 0.01322 ± 0.00054259 (n=10) | 24.365 |
| MNIST (exploratory) | receptive_field | 2000 | 500 | activity | 0.93731 ± 0.00078223 (n=10) | 0.01349 ± 0.0017927 (n=10) | 7.5248 |
| MNIST (exploratory) | receptive_field | 2000 | 500 | magnitude | 0.94386 ± 0.00074762 (n=10) | 0.02004 ± 0.0017204 (n=10) | 11.648 |
| MNIST (exploratory) | receptive_field | 2000 | 500 | obs | 0.95404 ± 0.00037452 (n=10) | 0.03022 ± 0.0015715 (n=10) | 19.23 |
| MNIST (exploratory) | receptive_field | 2000 | 500 | obs_oneshot | 0.95062 ± 0.0005046 (n=10) | 0.0268 ± 0.001573 (n=10) | 17.038 |
| MNIST (exploratory) | receptive_field | 2000 | 250 | activity | 0.90468 ± 0.00071985 (n=10) | 0.01602 ± 0.0029407 (n=10) | 5.4476 |
| MNIST (exploratory) | receptive_field | 2000 | 250 | magnitude | 0.91251 ± 0.0011299 (n=10) | 0.02385 ± 0.0031817 (n=10) | 7.4961 |
| MNIST (exploratory) | receptive_field | 2000 | 250 | obs | 0.94078 ± 0.00050789 (n=10) | 0.05212 ± 0.0028062 (n=10) | 18.573 |
| MNIST (exploratory) | receptive_field | 2000 | 250 | obs_oneshot | 0.92865 ± 0.0011567 (n=10) | 0.03999 ± 0.0025693 (n=10) | 15.564 |
| MNIST (exploratory) | receptive_field | 4000 | 4000 | pool | 0.97092 ± 0.00043071 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | 4000 | 2000 | activity | 0.9684 ± 0.00048534 (n=10) | 0.00562 ± 0.0005035 (n=10) | 11.162 |
| MNIST (exploratory) | receptive_field | 4000 | 2000 | magnitude | 0.96932 ± 0.00033757 (n=10) | 0.00654 ± 0.00039699 (n=10) | 16.474 |
| MNIST (exploratory) | receptive_field | 4000 | 2000 | obs | 0.9703 ± 0.00025342 (n=10) | 0.00752 ± 0.00025768 (n=10) | 29.183 |
| MNIST (exploratory) | receptive_field | 4000 | 2000 | obs_oneshot | 0.96992 ± 0.00032992 (n=10) | 0.00714 ± 0.00034196 (n=10) | 20.88 |
| MNIST (exploratory) | receptive_field | 4000 | 1000 | activity | 0.95735 ± 0.00034359 (n=10) | 0.00963 ± 0.00068086 (n=10) | 14.144 |
| MNIST (exploratory) | receptive_field | 4000 | 1000 | magnitude | 0.96045 ± 0.00046314 (n=10) | 0.01273 ± 0.00064326 (n=10) | 19.79 |
| MNIST (exploratory) | receptive_field | 4000 | 1000 | obs | 0.96599 ± 0.0003746 (n=10) | 0.01827 ± 0.00074536 (n=10) | 24.512 |
| MNIST (exploratory) | receptive_field | 4000 | 1000 | obs_oneshot | 0.96484 ± 0.00036399 (n=10) | 0.01712 ± 0.00055112 (n=10) | 31.064 |
| MNIST (exploratory) | receptive_field | 4000 | 500 | activity | 0.93438 ± 0.00054012 (n=10) | 0.01056 ± 0.0015084 (n=10) | 7.0006 |
| MNIST (exploratory) | receptive_field | 4000 | 500 | magnitude | 0.93855 ± 0.00040859 (n=10) | 0.01473 ± 0.0015065 (n=10) | 9.7776 |
| MNIST (exploratory) | receptive_field | 4000 | 500 | obs | 0.95893 ± 0.00028946 (n=10) | 0.03511 ± 0.0013767 (n=10) | 25.502 |
| MNIST (exploratory) | receptive_field | 4000 | 500 | obs_oneshot | 0.95139 ± 0.00050628 (n=10) | 0.02757 ± 0.0014871 (n=10) | 18.54 |
| MNIST (exploratory) | receptive_field | 4000 | 250 | activity | 0.89927 ± 0.0012425 (n=10) | 0.01061 ± 0.0025228 (n=10) | 4.2056 |
| MNIST (exploratory) | receptive_field | 4000 | 250 | magnitude | 0.90096 ± 0.0010329 (n=10) | 0.0123 ± 0.002383 (n=10) | 5.1616 |
| MNIST (exploratory) | receptive_field | 4000 | 250 | obs | 0.94596 ± 0.00034033 (n=10) | 0.0573 ± 0.0024722 (n=10) | 23.178 |
| MNIST (exploratory) | receptive_field | 4000 | 250 | obs_oneshot | 0.92649 ± 0.0010114 (n=10) | 0.03783 ± 0.0028447 (n=10) | 13.298 |
| MNIST (exploratory) | receptive_field | 8000 | 8000 | pool | 0.97651 ± 0.00024696 (n=10) | 0 ± 0 (n=10) | 0 |
| MNIST (exploratory) | receptive_field | 8000 | 2000 | activity | 0.96888 ± 0.00051764 (n=10) | 0.0061 ± 0.00051403 (n=10) | 11.867 |
| MNIST (exploratory) | receptive_field | 8000 | 2000 | magnitude | 0.97037 ± 0.00028946 (n=10) | 0.00759 ± 0.00041 (n=10) | 18.512 |
| MNIST (exploratory) | receptive_field | 8000 | 2000 | obs | 0.97326 ± 0.00022618 (n=10) | 0.01048 ± 0.00043507 (n=10) | 24.088 |
| MNIST (exploratory) | receptive_field | 8000 | 2000 | obs_oneshot | 0.9728 ± 0.00029477 (n=10) | 0.01002 ± 0.0002962 (n=10) | 33.829 |
| MNIST (exploratory) | receptive_field | 8000 | 1000 | activity | 0.95491 ± 0.00041216 (n=10) | 0.00719 ± 0.00059823 (n=10) | 12.019 |
| MNIST (exploratory) | receptive_field | 8000 | 1000 | magnitude | 0.95671 ± 0.00044358 (n=10) | 0.00899 ± 0.00065412 (n=10) | 13.744 |
| MNIST (exploratory) | receptive_field | 8000 | 1000 | obs | 0.96859 ± 0.00031709 (n=10) | 0.02087 ± 0.00061047 (n=10) | 34.187 |
| MNIST (exploratory) | receptive_field | 8000 | 1000 | obs_oneshot | 0.96403 ± 0.00037 (n=10) | 0.01631 ± 0.00064661 (n=10) | 25.224 |
| MNIST (exploratory) | receptive_field | 8000 | 500 | activity | 0.92944 ± 0.00079529 (n=10) | 0.00562 ± 0.0012448 (n=10) | 4.5148 |
| MNIST (exploratory) | receptive_field | 8000 | 500 | magnitude | 0.93023 ± 0.00080375 (n=10) | 0.00641 ± 0.0015263 (n=10) | 4.1996 |
| MNIST (exploratory) | receptive_field | 8000 | 500 | obs | 0.96093 ± 0.00037 (n=10) | 0.03711 ± 0.0014012 (n=10) | 26.484 |
| MNIST (exploratory) | receptive_field | 8000 | 500 | obs_oneshot | 0.9484 ± 0.00059029 (n=10) | 0.02458 ± 0.001562 (n=10) | 15.736 |
| MNIST (exploratory) | receptive_field | 8000 | 250 | activity | 0.89204 ± 0.0012827 (n=10) | 0.00338 ± 0.0019841 (n=10) | 1.7035 |
| MNIST (exploratory) | receptive_field | 8000 | 250 | magnitude | 0.89478 ± 0.00082149 (n=10) | 0.00612 ± 0.0025267 (n=10) | 2.4221 |
| MNIST (exploratory) | receptive_field | 8000 | 250 | obs | 0.94918 ± 0.00049414 (n=10) | 0.06052 ± 0.0025576 (n=10) | 23.663 |
| MNIST (exploratory) | receptive_field | 8000 | 250 | obs_oneshot | 0.9222 ± 0.00074072 (n=10) | 0.03354 ± 0.002942 (n=10) | 11.4 |

### Growth

| Dataset | Family | k | Arm | Gen 0 | Final | Val-selected | Final Δ random | Equivalent width | Mean test by generation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FashionMNIST | global | 250 | agop | 0.80708 ± 0.0010718 (n=10) | 0.83451 ± 0.00092789 (n=10) | 0.83333 ± 0.00085298 (n=10) | 0.00298 ± 0.00093972 (n=10) | 550.6 | 0:0.80708, 1:0.82248, 2:0.82663, 3:0.82886, 4:0.83006, 5:0.83146, 6:0.83162, 7:0.83209, 8:0.83336, 9:0.83373, 10:0.83364, 11:0.83413, 12:0.83451 |
| FashionMNIST | global | 250 | first_order | 0.80708 ± 0.0010718 (n=10) | 0.84201 ± 0.00070118 (n=10) | 0.84169 ± 0.00078038 (n=10) | 0.01048 ± 0.00098903 (n=10) | 748.6 | 0:0.80708, 1:0.82557, 2:0.83292, 3:0.8362, 4:0.83782, 5:0.83919, 6:0.83948, 7:0.84047, 8:0.84088, 9:0.84099, 10:0.84133, 11:0.84152, 12:0.84201 |
| FashionMNIST | global | 250 | pca | 0.80708 ± 0.0010718 (n=10) | 0.83284 ± 0.00051752 (n=10) | 0.83241 ± 0.00050408 (n=10) | 0.00131 ± 0.00072425 (n=10) | 532.2 | 0:0.80708, 1:0.82097, 2:0.8236, 3:0.82639, 4:0.82755, 5:0.82833, 6:0.82928, 7:0.83053, 8:0.83042, 9:0.83089, 10:0.83239, 11:0.83274, 12:0.83284 |
| FashionMNIST | global | 250 | phd | 0.80708 ± 0.0010718 (n=10) | 0.85551 ± 0.00056734 (n=10) | 0.85521 ± 0.00054721 (n=10) | 0.02398 ± 0.00062606 (n=10) | 1375.2 | 0:0.80708, 1:0.8341, 2:0.84359, 3:0.84794, 4:0.84932, 5:0.85143, 6:0.85219, 7:0.85317, 8:0.85372, 9:0.85487, 10:0.85466, 11:0.85576, 12:0.85551 |
| FashionMNIST | global | 250 | phd_classshuffle | 0.80708 ± 0.0010718 (n=10) | 0.84022 ± 0.00050129 (n=10) | 0.84011 ± 0.00053322 (n=10) | 0.00869 ± 0.0010598 (n=10) | 706.4 | 0:0.80708, 1:0.82598, 2:0.83006, 3:0.83222, 4:0.83296, 5:0.83447, 6:0.83624, 7:0.83763, 8:0.83923, 9:0.83952, 10:0.83943, 11:0.84072, 12:0.84022 |
| FashionMNIST | global | 250 | phd_frozen | 0.80708 ± 0.0010718 (n=10) | 0.85082 ± 0.00044667 (n=10) | 0.85003 ± 0.00052958 (n=10) | 0.01929 ± 0.00093148 (n=10) | 1061.8 | 0:0.80708, 1:0.8341, 2:0.84024, 3:0.84346, 4:0.84607, 5:0.84711, 6:0.84734, 7:0.84866, 8:0.84894, 9:0.84968, 10:0.84984, 11:0.85033, 12:0.85082 |
| FashionMNIST | global | 250 | phd_randkeep | 0.80708 ± 0.0010718 (n=10) | 0.83573 ± 0.00048512 (n=10) | 0.83669 ± 0.000626 (n=10) | 0.0042 ± 0.00097365 (n=10) | 622.9 | 0:0.80708, 1:0.82327, 2:0.83013, 3:0.83276, 4:0.83293, 5:0.83506, 6:0.83474, 7:0.83615, 8:0.83565, 9:0.83515, 10:0.83532, 11:0.83486, 12:0.83573 |
| FashionMNIST | global | 250 | random | 0.80708 ± 0.0010718 (n=10) | 0.83153 ± 0.00067 (n=10) | 0.83119 ± 0.00063218 (n=10) | 0 ± 0 (n=10) | 508.9 | 0:0.80708, 1:0.81902, 2:0.82316, 3:0.8248, 4:0.82611, 5:0.82695, 6:0.82844, 7:0.83027, 8:0.8306, 9:0.83066, 10:0.83136, 11:0.83186, 12:0.83153 |
| FashionMNIST | global | 1000 | agop | 0.84736 ± 0.00063108 (n=10) | 0.86676 ± 0.00049062 (n=10) | 0.86696 ± 0.00050403 (n=10) | 0.00299 ± 0.00059094 (n=10) | 2600.1 | 0:0.84736, 1:0.85833, 2:0.86153, 3:0.86361, 4:0.86444, 5:0.865, 6:0.8648, 7:0.86572, 8:0.86621, 9:0.86624, 10:0.86632, 11:0.86686, 12:0.86676 |
| FashionMNIST | global | 1000 | first_order | 0.84736 ± 0.00063108 (n=10) | 0.85734 ± 0.00045417 (n=10) | 0.85724 ± 0.0006085 (n=10) | -0.00643 ± 0.00054651 (n=10) | 1523.5 | 0:0.84736, 1:0.8527, 2:0.85402, 3:0.85503, 4:0.85529, 5:0.85591, 6:0.85619, 7:0.85661, 8:0.85688, 9:0.8569, 10:0.8569, 11:0.85725, 12:0.85734 |
| FashionMNIST | global | 1000 | pca | 0.84736 ± 0.00063108 (n=10) | 0.86182 ± 0.00041253 (n=10) | 0.86074 ± 0.00045319 (n=10) | -0.00195 ± 0.00079711 (n=10) | 1837.7 | 0:0.84736, 1:0.85508, 2:0.85618, 3:0.85844, 4:0.85817, 5:0.85875, 6:0.85943, 7:0.85958, 8:0.86058, 9:0.86052, 10:0.86088, 11:0.86124, 12:0.86182 |
| FashionMNIST | global | 1000 | phd | 0.84736 ± 0.00063108 (n=10) | 0.88031 ± 0.00045153 (n=10) | 0.87998 ± 0.0004276 (n=10) | 0.01654 ± 0.00084264 (n=10) | 5984.8 | 0:0.84736, 1:0.86671, 2:0.8717, 3:0.87424, 4:0.87612, 5:0.877, 6:0.8775, 7:0.87851, 8:0.87878, 9:0.8793, 10:0.87999, 11:0.87993, 12:0.88031 |
| FashionMNIST | global | 1000 | phd_classshuffle | 0.84736 ± 0.00063108 (n=10) | 0.86714 ± 0.00051212 (n=10) | 0.86716 ± 0.00056257 (n=10) | 0.00337 ± 0.00075808 (n=10) | 2629.6 | 0:0.84736, 1:0.85758, 2:0.85977, 3:0.86216, 4:0.86269, 5:0.86395, 6:0.86531, 7:0.86568, 8:0.8661, 9:0.86619, 10:0.86646, 11:0.86692, 12:0.86714 |
| FashionMNIST | global | 1000 | phd_frozen | 0.84736 ± 0.00063108 (n=10) | 0.87836 ± 0.00046481 (n=10) | 0.87789 ± 0.00048499 (n=10) | 0.01459 ± 0.00080173 (n=10) | 5240.4 | 0:0.84736, 1:0.86671, 2:0.87085, 3:0.87329, 4:0.87398, 5:0.87432, 6:0.87522, 7:0.87602, 8:0.87646, 9:0.87747, 10:0.87745, 11:0.87765, 12:0.87836 |
| FashionMNIST | global | 1000 | phd_randkeep | 0.84736 ± 0.00063108 (n=10) | 0.86555 ± 0.00068171 (n=10) | 0.86535 ± 0.00050755 (n=10) | 0.00178 ± 0.00092997 (n=10) | 2373.8 | 0:0.84736, 1:0.85963, 2:0.86204, 3:0.86447, 4:0.86484, 5:0.86359, 6:0.86429, 7:0.8639, 8:0.86516, 9:0.86527, 10:0.8656, 11:0.86556, 12:0.86555 |
| FashionMNIST | global | 1000 | random | 0.84736 ± 0.00063108 (n=10) | 0.86377 ± 0.00050773 (n=10) | 0.86358 ± 0.00044642 (n=10) | 0 ± 0 (n=10) | 2147.7 | 0:0.84736, 1:0.85444, 2:0.85725, 3:0.85854, 4:0.85996, 5:0.86187, 6:0.86217, 7:0.86308, 8:0.8631, 9:0.86357, 10:0.86391, 11:0.86406, 12:0.86377 |
| FashionMNIST | patch | 250 | phd_patch | 0.80536 ± 0.0011483 (n=10) | 0.84664 ± 0.00057797 (n=10) | 0.84623 ± 0.0007338 (n=10) | 0.00776 ± 0.00065629 (n=10) | 843.1 | 0:0.80536, 1:0.82761, 2:0.83504, 3:0.83841, 4:0.84069, 5:0.84191, 6:0.84306, 7:0.84393, 8:0.84486, 9:0.8454, 10:0.84633, 11:0.84546, 12:0.84664 |
| FashionMNIST | patch | 250 | random | 0.80536 ± 0.0011483 (n=10) | 0.83888 ± 0.00046495 (n=10) | 0.83808 ± 0.0005748 (n=10) | 0 ± 0 (n=10) | 643.8 | 0:0.80536, 1:0.82054, 2:0.82659, 3:0.82946, 4:0.83168, 5:0.83273, 6:0.83344, 7:0.83536, 8:0.83607, 9:0.83752, 10:0.83815, 11:0.83802, 12:0.83888 |
| FashionMNIST | patch | 1000 | phd_patch | 0.85062 ± 0.00059569 (n=10) | 0.87939 ± 0.00046175 (n=10) | 0.87945 ± 0.00045 (n=10) | 0.00524 ± 0.00054185 (n=10) | 3510.4 | 0:0.85062, 1:0.86596, 2:0.87127, 3:0.87397, 4:0.87557, 5:0.87574, 6:0.87711, 7:0.8775, 8:0.87788, 9:0.8781, 10:0.87881, 11:0.87948, 12:0.87939 |
| FashionMNIST | patch | 1000 | random | 0.85062 ± 0.00059569 (n=10) | 0.87415 ± 0.0004285 (n=10) | 0.87338 ± 0.00074458 (n=10) | 0 ± 0 (n=10) | 2512.2 | 0:0.85062, 1:0.86171, 2:0.86508, 3:0.86733, 4:0.86894, 5:0.87058, 6:0.87206, 7:0.87307, 8:0.87327, 9:0.8734, 10:0.87368, 11:0.87336, 12:0.87415 |
| CIFAR10 | global | 250 | phd | 0.40293 ± 0.00080209 (n=10) | 0.46425 ± 0.0012747 (n=10) | 0.46283 ± 0.0011521 (n=10) | 0.03103 ± 0.0016749 (n=10) | 1244.2 | 0:0.40293, 1:0.43516, 2:0.44555, 3:0.45167, 4:0.45525, 5:0.45711, 6:0.45879, 7:0.45972, 8:0.46056, 9:0.46225, 10:0.46342, 11:0.46407, 12:0.46425 |
| CIFAR10 | global | 250 | phd_classshuffle | 0.40293 ± 0.00080209 (n=10) | 0.45173 ± 0.0010323 (n=10) | 0.45123 ± 0.0011585 (n=10) | 0.01851 ± 0.00078152 (n=10) | 873.3 | 0:0.40293, 1:0.42848, 2:0.43499, 3:0.43862, 4:0.44118, 5:0.44324, 6:0.44627, 7:0.44727, 8:0.448, 9:0.44899, 10:0.44905, 11:0.45099, 12:0.45173 |
| CIFAR10 | global | 250 | phd_randkeep | 0.40293 ± 0.00080209 (n=10) | 0.43037 ± 0.00099119 (n=10) | 0.43359 ± 0.00077752 (n=10) | -0.00285 ± 0.0016657 (n=10) | 519.8 | 0:0.40293, 1:0.42116, 2:0.42792, 3:0.43165, 4:0.43395, 5:0.43258, 6:0.43002, 7:0.43225, 8:0.43193, 9:0.43167, 10:0.43206, 11:0.43086, 12:0.43037 |
| CIFAR10 | global | 250 | random | 0.40293 ± 0.00080209 (n=10) | 0.43322 ± 0.00098113 (n=10) | 0.43218 ± 0.0013111 (n=10) | 0 ± 0 (n=10) | 498.9 | 0:0.40293, 1:0.41786, 2:0.42136, 3:0.42521, 4:0.42668, 5:0.42873, 6:0.43061, 7:0.4293, 8:0.43052, 9:0.43138, 10:0.4324, 11:0.43355, 12:0.43322 |
| CIFAR10 | global | 1000 | phd | 0.45515 ± 0.00088785 (n=10) | 0.5018 ± 0.00089094 (n=10) | 0.50082 ± 0.00075407 (n=10) | 0.02268 ± 0.0010181 (n=10) | 6566.2 | 0:0.45515, 1:0.48427, 2:0.49261, 3:0.4954, 4:0.49603, 5:0.49723, 6:0.49744, 7:0.49847, 8:0.49849, 9:0.4993, 10:0.50017, 11:0.50218, 12:0.5018 |
| CIFAR10 | global | 1000 | phd_classshuffle | 0.45515 ± 0.00088785 (n=10) | 0.49154 ± 0.00077305 (n=10) | 0.49028 ± 0.00071411 (n=10) | 0.01242 ± 0.00095949 (n=10) | 3566.4 | 0:0.45515, 1:0.47307, 2:0.47729, 3:0.48136, 4:0.48223, 5:0.48481, 6:0.48644, 7:0.48663, 8:0.4873, 9:0.48926, 10:0.48978, 11:0.48965, 12:0.49154 |
| CIFAR10 | global | 1000 | phd_randkeep | 0.45515 ± 0.00088785 (n=10) | 0.47783 ± 0.00092365 (n=10) | 0.4783 ± 0.00089876 (n=10) | -0.00129 ± 0.0014615 (n=10) | 2062.5 | 0:0.45515, 1:0.47334, 2:0.47837, 3:0.47957, 4:0.48003, 5:0.47916, 6:0.47931, 7:0.48066, 8:0.47851, 9:0.48014, 10:0.47722, 11:0.48012, 12:0.47783 |
| CIFAR10 | global | 1000 | random | 0.45515 ± 0.00088785 (n=10) | 0.47912 ± 0.00091004 (n=10) | 0.47786 ± 0.00067267 (n=10) | 0 ± 0 (n=10) | 2024.6 | 0:0.45515, 1:0.46821, 2:0.46911, 3:0.47156, 4:0.47207, 5:0.47316, 6:0.47436, 7:0.47609, 8:0.4763, 9:0.47745, 10:0.47858, 11:0.47879, 12:0.47912 |
| CIFAR10 | patch | 250 | phd_patch | 0.42163 ± 0.0012864 (n=10) | 0.4813 ± 0.0017252 (n=10) | 0.48144 ± 0.0015341 (n=10) | 0.00568 ± 0.0018828 (n=10) | 828.8 | 0:0.42163, 1:0.44957, 2:0.46005, 3:0.46572, 4:0.47055, 5:0.47326, 6:0.47512, 7:0.4767, 8:0.47929, 9:0.48017, 10:0.48013, 11:0.48047, 12:0.4813 |
| CIFAR10 | patch | 250 | random | 0.42163 ± 0.0012864 (n=10) | 0.47562 ± 0.00092 (n=10) | 0.47468 ± 0.00075672 (n=10) | 0 ± 0 (n=10) | 716.2 | 0:0.42163, 1:0.4424, 2:0.4509, 3:0.45818, 4:0.46078, 5:0.46453, 6:0.46643, 7:0.47006, 8:0.47101, 9:0.47153, 10:0.47351, 11:0.47515, 12:0.47562 |
| CIFAR10 | patch | 1000 | phd_patch | 0.48651 ± 0.001831 (n=10) | 0.54039 ± 0.00092778 (n=10) | 0.53956 ± 0.00088734 (n=10) | 0.00517 ± 0.0014527 (n=10) | 3315.7 | 0:0.48651, 1:0.5114, 2:0.51965, 3:0.52369, 4:0.5276, 5:0.53093, 6:0.53229, 7:0.53393, 8:0.53602, 9:0.53734, 10:0.53809, 11:0.53881, 12:0.54039 |
| CIFAR10 | patch | 1000 | random | 0.48651 ± 0.001831 (n=10) | 0.53522 ± 0.0012517 (n=10) | 0.53517 ± 0.001246 (n=10) | 0 ± 0 (n=10) | 2920.9 | 0:0.48651, 1:0.50777, 2:0.51502, 3:0.52053, 4:0.52402, 5:0.52669, 6:0.52933, 7:0.52956, 8:0.53127, 9:0.53264, 10:0.53283, 11:0.53408, 12:0.53522 |
| MNIST (exploratory) | global | 250 | agop | 0.8907 ± 0.00098658 (n=10) | 0.92554 ± 0.00067036 (n=10) | 0.92507 ± 0.00065218 (n=10) | 9e-05 ± 0.00077624 (n=10) | 526.8 | 0:0.8907, 1:0.91108, 2:0.91754, 3:0.91986, 4:0.92126, 5:0.92307, 6:0.9238, 7:0.92455, 8:0.92469, 9:0.92494, 10:0.92496, 11:0.92466, 12:0.92554 |
| MNIST (exploratory) | global | 250 | first_order | 0.8907 ± 0.00098658 (n=10) | 0.92033 ± 0.0009454 (n=10) | 0.91993 ± 0.0009536 (n=10) | -0.00512 ± 0.00080868 (n=10) | 465 | 0:0.8907, 1:0.90446, 2:0.91059, 3:0.91287, 4:0.91423, 5:0.91615, 6:0.91692, 7:0.91786, 8:0.91881, 9:0.91891, 10:0.9195, 11:0.92012, 12:0.92033 |
| MNIST (exploratory) | global | 250 | pca | 0.8907 ± 0.00098658 (n=10) | 0.94852 ± 0.00038609 (n=10) | 0.94821 ± 0.00035448 (n=10) | 0.02307 ± 0.00035435 (n=10) | 1208.3 | 0:0.8907, 1:0.93281, 2:0.94122, 3:0.94328, 4:0.94493, 5:0.94553, 6:0.94669, 7:0.94706, 8:0.94812, 9:0.94801, 10:0.94779, 11:0.94833, 12:0.94852 |
| MNIST (exploratory) | global | 250 | phd | 0.8907 ± 0.00098658 (n=10) | 0.95428 ± 0.00039294 (n=10) | 0.95366 ± 0.00062879 (n=10) | 0.02883 ± 0.00038616 (n=10) | 1560.7 | 0:0.8907, 1:0.93638, 2:0.94461, 3:0.94816, 4:0.94893, 5:0.95018, 6:0.95087, 7:0.95169, 8:0.95223, 9:0.9528, 10:0.9537, 11:0.95427, 12:0.95428 |
| MNIST (exploratory) | global | 250 | phd_classshuffle | 0.8907 ± 0.00098658 (n=10) | 0.94915 ± 0.00058921 (n=10) | 0.9485 ± 0.00043102 (n=10) | 0.0237 ± 0.00059647 (n=10) | 1224.7 | 0:0.8907, 1:0.93588, 2:0.94231, 3:0.94419, 4:0.94581, 5:0.94656, 6:0.94702, 7:0.94803, 8:0.94802, 9:0.94828, 10:0.9491, 11:0.94915, 12:0.94915 |
| MNIST (exploratory) | global | 250 | phd_frozen | 0.8907 ± 0.00098658 (n=10) | 0.95172 ± 0.0005312 (n=10) | 0.95122 ± 0.00052192 (n=10) | 0.02627 ± 0.00046858 (n=10) | 1389.9 | 0:0.8907, 1:0.93638, 2:0.94457, 3:0.94699, 4:0.94838, 5:0.94893, 6:0.94933, 7:0.94985, 8:0.95033, 9:0.95072, 10:0.95098, 11:0.95149, 12:0.95172 |
| MNIST (exploratory) | global | 250 | phd_randkeep | 0.8907 ± 0.00098658 (n=10) | 0.93561 ± 0.0004981 (n=10) | 0.9354 ± 0.00065929 (n=10) | 0.01016 ± 0.00071308 (n=10) | 724.5 | 0:0.8907, 1:0.92203, 2:0.93081, 3:0.9314, 4:0.93378, 5:0.9335, 6:0.93404, 7:0.93388, 8:0.93542, 9:0.93444, 10:0.93522, 11:0.93465, 12:0.93561 |
| MNIST (exploratory) | global | 250 | random | 0.8907 ± 0.00098658 (n=10) | 0.92545 ± 0.00033074 (n=10) | 0.9257 ± 0.00026708 (n=10) | 0 ± 0 (n=10) | 537.1 | 0:0.8907, 1:0.90786, 2:0.91278, 3:0.91601, 4:0.91727, 5:0.91985, 6:0.92072, 7:0.92192, 8:0.92271, 9:0.92367, 10:0.92451, 11:0.92543, 12:0.92545 |
| MNIST (exploratory) | global | 1000 | agop | 0.94426 ± 0.00038361 (n=10) | 0.96226 ± 0.00022764 (n=10) | 0.962 ± 0.00029439 (n=10) | 0.00133 ± 0.00040772 (n=10) | 2579.8 | 0:0.94426, 1:0.95408, 2:0.95702, 3:0.95858, 4:0.95926, 5:0.96024, 6:0.96051, 7:0.96112, 8:0.96138, 9:0.96183, 10:0.96174, 11:0.96206, 12:0.96226 |
| MNIST (exploratory) | global | 1000 | first_order | 0.94426 ± 0.00038361 (n=10) | 0.94871 ± 0.00058127 (n=10) | 0.94877 ± 0.0004951 (n=10) | -0.01222 ± 0.00063523 (n=10) | 1240.2 | 0:0.94426, 1:0.94626, 2:0.94693, 3:0.94719, 4:0.9474, 5:0.94776, 6:0.94822, 7:0.94851, 8:0.94829, 9:0.9483, 10:0.94876, 11:0.94871, 12:0.94871 |
| MNIST (exploratory) | global | 1000 | pca | 0.94426 ± 0.00038361 (n=10) | 0.97105 ± 0.00020125 (n=10) | 0.97047 ± 0.0002757 (n=10) | 0.01012 ± 0.00043914 (n=10) | 5235.4 | 0:0.94426, 1:0.96285, 2:0.96592, 3:0.96808, 4:0.96821, 5:0.9691, 6:0.96956, 7:0.97017, 8:0.97002, 9:0.97037, 10:0.97062, 11:0.97072, 12:0.97105 |
| MNIST (exploratory) | global | 1000 | phd | 0.94426 ± 0.00038361 (n=10) | 0.97381 ± 0.00034943 (n=10) | 0.97357 ± 0.00049442 (n=10) | 0.01288 ± 0.00048116 (n=10) | 7908.9 | 0:0.94426, 1:0.96571, 2:0.96973, 3:0.97119, 4:0.97238, 5:0.97228, 6:0.97271, 7:0.97302, 8:0.97356, 9:0.97354, 10:0.97365, 11:0.97382, 12:0.97381 |
| MNIST (exploratory) | global | 1000 | phd_classshuffle | 0.94426 ± 0.00038361 (n=10) | 0.97116 ± 0.00033606 (n=10) | 0.97107 ± 0.00029928 (n=10) | 0.01023 ± 0.00046953 (n=10) | 5614.5 | 0:0.94426, 1:0.96316, 2:0.9671, 3:0.96795, 4:0.96922, 5:0.96982, 6:0.96958, 7:0.97003, 8:0.97042, 9:0.97045, 10:0.97081, 11:0.97113, 12:0.97116 |
| MNIST (exploratory) | global | 1000 | phd_frozen | 0.94426 ± 0.00038361 (n=10) | 0.97412 ± 0.00032069 (n=10) | 0.97384 ± 0.00048744 (n=10) | 0.01319 ± 0.00043728 (n=10) | >8000 | 0:0.94426, 1:0.96571, 2:0.96922, 3:0.97108, 4:0.97241, 5:0.97262, 6:0.97258, 7:0.97342, 8:0.97362, 9:0.974, 10:0.97397, 11:0.97389, 12:0.97412 |
| MNIST (exploratory) | global | 1000 | phd_randkeep | 0.94426 ± 0.00038361 (n=10) | 0.96428 ± 0.00038262 (n=10) | 0.9652 ± 0.00052026 (n=10) | 0.00335 ± 0.00065917 (n=10) | 3262.8 | 0:0.94426, 1:0.95887, 2:0.96275, 3:0.96371, 4:0.96402, 5:0.96414, 6:0.96486, 7:0.96509, 8:0.96533, 9:0.96473, 10:0.96399, 11:0.96426, 12:0.96428 |
| MNIST (exploratory) | global | 1000 | random | 0.94426 ± 0.00038361 (n=10) | 0.96093 ± 0.00038616 (n=10) | 0.96082 ± 0.00045505 (n=10) | 0 ± 0 (n=10) | 2357.9 | 0:0.94426, 1:0.95198, 2:0.95477, 3:0.95604, 4:0.95759, 5:0.95783, 6:0.95847, 7:0.95889, 8:0.95902, 9:0.95967, 10:0.96006, 11:0.96072, 12:0.96093 |
| MNIST (exploratory) | patch | 250 | phd_patch | 0.88777 ± 0.001418 (n=10) | 0.95546 ± 0.00041931 (n=10) | 0.95516 ± 0.00033606 (n=10) | 0.01137 ± 0.0008296 (n=10) | 1359.6 | 0:0.88777, 1:0.93002, 2:0.94035, 3:0.9454, 4:0.9486, 5:0.94949, 6:0.95085, 7:0.952, 8:0.95255, 9:0.95374, 10:0.95385, 11:0.95506, 12:0.95546 |
| MNIST (exploratory) | patch | 250 | random | 0.88777 ± 0.001418 (n=10) | 0.94409 ± 0.0005986 (n=10) | 0.94405 ± 0.000619 (n=10) | 0 ± 0 (n=10) | 875 | 0:0.88777, 1:0.91798, 2:0.92555, 3:0.93146, 4:0.93493, 5:0.93662, 6:0.93901, 7:0.94074, 8:0.94215, 9:0.94229, 10:0.9433, 11:0.94412, 12:0.94409 |
| MNIST (exploratory) | patch | 1000 | phd_patch | 0.94755 ± 0.00056887 (n=10) | 0.97504 ± 0.00023343 (n=10) | 0.97447 ± 0.0004107 (n=10) | 0.00544 ± 0.00039277 (n=10) | 6148.9 | 0:0.94755, 1:0.96524, 2:0.97002, 3:0.97142, 4:0.97215, 5:0.97299, 6:0.97312, 7:0.97364, 8:0.97366, 9:0.97432, 10:0.97417, 11:0.97485, 12:0.97504 |
| MNIST (exploratory) | patch | 1000 | random | 0.94755 ± 0.00056887 (n=10) | 0.9696 ± 0.00037298 (n=10) | 0.9695 ± 0.00042921 (n=10) | 0 ± 0 (n=10) | 3436.4 | 0:0.94755, 1:0.96043, 2:0.96396, 3:0.96558, 4:0.96663, 5:0.96805, 6:0.96805, 7:0.96858, 8:0.9689, 9:0.96911, 10:0.96927, 11:0.96939, 12:0.9696 |

### Primary paired contrasts: `test_accuracy` (validation-selected alpha); final generation for growth

| Dataset | Family/strategy | P | k | Contrast | n | Mean Δ | SE | t | One-sided p | Unadjusted lower 95% | Holm p | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FashionMNIST | hypersphere | 4000 | 1000 | obs − random | 10 | 0.01166 | 0.00054369 | 21.446 | 2.4507e-09 | 0.010663 | — | — |
| FashionMNIST | receptive_field | 4000 | 1000 | obs − random | 10 | 0.01811 | 0.0011521 | 15.719 | 3.7545e-08 | 0.015998 | — | — |
| FashionMNIST | global | — | 250 | width3x | 10 | 0.01377 | 0.00079359 | 17.351 | 1.583e-08 | 0.012315 | — | — |
| FashionMNIST | global | — | 250 | proposal | 10 | 0.02398 | 0.00062606 | 38.303 | 1.3993e-11 | 0.022832 | — | — |
| FashionMNIST | global | — | 250 | mechanism | 10 | 0.01529 | 0.00086531 | 17.67 | 1.3497e-08 | 0.013704 | — | — |
| FashionMNIST | global | — | 250 | selection | 10 | 0.01978 | 0.00080758 | 24.493 | 7.5508e-10 | 0.0183 | — | — |
| FashionMNIST | global | — | 1000 | width3x | 10 | 0.01082 | 0.00090772 | 11.92 | 4.0757e-07 | 0.009156 | 8.1513e-07 | pass |
| FashionMNIST | global | — | 1000 | proposal | 10 | 0.01654 | 0.00084264 | 19.629 | 5.3552e-09 | 0.014995 | 2.6776e-08 | pass |
| FashionMNIST | global | — | 1000 | mechanism | 10 | 0.01317 | 0.00071849 | 18.33 | 9.7814e-09 | 0.011853 | 3.9126e-08 | pass |
| FashionMNIST | global | — | 1000 | selection | 10 | 0.01476 | 0.00093145 | 15.846 | 3.4988e-08 | 0.013053 | — | — |
| FashionMNIST | patch | — | 250 | width3x | 10 | 0.00372 | 0.0011863 | 3.1358 | 0.0060043 | 0.0015454 | — | — |
| FashionMNIST | patch | — | 250 | proposal | 10 | 0.00776 | 0.00065629 | 11.824 | 4.3655e-07 | 0.006557 | — | — |
| FashionMNIST | patch | — | 1000 | width3x | 10 | 0.00278 | 0.00049279 | 5.6413 | 0.00015854 | 0.0018767 | — | — |
| FashionMNIST | patch | — | 1000 | proposal | 10 | 0.00524 | 0.00054185 | 9.6706 | 2.3627e-06 | 0.0042467 | — | — |
| CIFAR10 | hypersphere | 4000 | 1000 | obs − random | 10 | 0.0148 | 0.0015304 | 9.6709 | 2.362e-06 | 0.011995 | — | — |
| CIFAR10 | receptive_field | 4000 | 1000 | obs − random | 10 | 0.03122 | 0.0016997 | 18.368 | 9.6034e-09 | 0.028104 | — | — |
| CIFAR10 | global | — | 250 | width3x | 10 | 0.01827 | 0.0016329 | 11.189 | 6.9716e-07 | 0.015277 | — | — |
| CIFAR10 | global | — | 250 | proposal | 10 | 0.03103 | 0.0016749 | 18.526 | 8.9073e-09 | 0.02796 | — | — |
| CIFAR10 | global | — | 250 | mechanism | 10 | 0.01252 | 0.001753 | 7.1422 | 2.7059e-05 | 0.0093066 | — | — |
| CIFAR10 | global | — | 250 | selection | 10 | 0.03388 | 0.0012946 | 26.171 | 4.1912e-10 | 0.031507 | — | — |
| CIFAR10 | global | — | 1000 | width3x | 10 | 0.01461 | 0.0012526 | 11.664 | 4.9025e-07 | 0.012314 | 8.1513e-07 | pass |
| CIFAR10 | global | — | 1000 | proposal | 10 | 0.02268 | 0.0010181 | 22.276 | 1.7515e-09 | 0.020814 | 1.0509e-08 | pass |
| CIFAR10 | global | — | 1000 | mechanism | 10 | 0.01026 | 0.00078361 | 13.093 | 1.8259e-07 | 0.0088236 | 5.4777e-07 | pass |
| CIFAR10 | global | — | 1000 | selection | 10 | 0.02397 | 0.0015636 | 15.33 | 4.6689e-08 | 0.021104 | — | — |
| CIFAR10 | patch | — | 250 | width3x | 10 | 0.0041 | 0.0017769 | 2.3074 | 0.023217 | 0.00084276 | — | — |
| CIFAR10 | patch | — | 250 | proposal | 10 | 0.00568 | 0.0018828 | 3.0167 | 0.0072781 | 0.0022286 | — | — |
| CIFAR10 | patch | — | 1000 | width3x | 10 | 0.00404 | 0.0010893 | 3.7089 | 0.0024265 | 0.0020433 | — | — |
| CIFAR10 | patch | — | 1000 | proposal | 10 | 0.00517 | 0.0014527 | 3.5588 | 0.0030654 | 0.002507 | — | — |
| MNIST (exploratory) | hypersphere | 4000 | 1000 | obs − random | 10 | 0.01227 | 0.00049756 | 24.66 | 7.1081e-10 | 0.011358 | — | — |
| MNIST (exploratory) | receptive_field | 4000 | 1000 | obs − random | 10 | 0.01827 | 0.00074536 | 24.512 | 7.5005e-10 | 0.016904 | — | — |
| MNIST (exploratory) | global | — | 250 | width3x | 10 | 0.01776 | 0.00050535 | 35.144 | 3.0219e-11 | 0.016834 | — | — |
| MNIST (exploratory) | global | — | 250 | proposal | 10 | 0.02883 | 0.00038616 | 74.658 | 3.51e-14 | 0.028122 | — | — |
| MNIST (exploratory) | global | — | 250 | mechanism | 10 | 0.00513 | 0.00063194 | 8.1179 | 9.8447e-06 | 0.0039716 | — | — |
| MNIST (exploratory) | global | — | 250 | selection | 10 | 0.01867 | 0.00081854 | 22.809 | 1.4205e-09 | 0.01717 | — | — |
| MNIST (exploratory) | global | — | 1000 | width3x | 10 | 0.00983 | 0.00041421 | 23.732 | 9.9917e-10 | 0.0090707 | — | — |
| MNIST (exploratory) | global | — | 1000 | proposal | 10 | 0.01288 | 0.00048116 | 26.769 | 3.4276e-10 | 0.011998 | — | — |
| MNIST (exploratory) | global | — | 1000 | mechanism | 10 | 0.00265 | 0.00047499 | 5.5791 | 0.00017167 | 0.0017793 | — | — |
| MNIST (exploratory) | global | — | 1000 | selection | 10 | 0.00953 | 0.00040989 | 23.25 | 1.1986e-09 | 0.0087786 | — | — |
| MNIST (exploratory) | patch | — | 250 | width3x | 10 | 0.01565 | 0.00096324 | 16.247 | 2.8135e-08 | 0.013884 | — | — |
| MNIST (exploratory) | patch | — | 250 | proposal | 10 | 0.01137 | 0.0008296 | 13.705 | 1.2321e-07 | 0.0098493 | — | — |
| MNIST (exploratory) | patch | — | 1000 | width3x | 10 | 0.00681 | 0.00029682 | 22.943 | 1.3483e-09 | 0.0062659 | — | — |
| MNIST (exploratory) | patch | — | 1000 | proposal | 10 | 0.00544 | 0.00039277 | 13.85 | 1.1251e-07 | 0.00472 | — | — |

### Secondary paired contrasts: `test_accuracy_loop` (fixed loop alpha); final generation for growth

| Dataset | Family/strategy | P | k | Contrast | n | Mean Δ | SE | t | One-sided p | Unadjusted lower 95% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FashionMNIST | hypersphere | 4000 | 1000 | obs − random | 10 | 0.0119 | 0.00056725 | 20.978 | 2.9782e-09 | 0.01086 |
| FashionMNIST | receptive_field | 4000 | 1000 | obs − random | 10 | 0.01855 | 0.0010948 | 16.944 | 1.9488e-08 | 0.016543 |
| FashionMNIST | global | — | 250 | width3x | 10 | 0.01248 | 0.00068326 | 18.265 | 1.009e-08 | 0.011228 |
| FashionMNIST | global | — | 250 | proposal | 10 | 0.02258 | 0.00049212 | 45.884 | 2.7756e-12 | 0.021678 |
| FashionMNIST | global | — | 250 | mechanism | 10 | 0.01699 | 0.00093719 | 18.129 | 1.0778e-08 | 0.015272 |
| FashionMNIST | global | — | 250 | selection | 10 | 0.02033 | 0.00081786 | 24.857 | 6.6227e-10 | 0.018831 |
| FashionMNIST | global | — | 1000 | width3x | 10 | 0.00942 | 0.00072123 | 13.061 | 1.865e-07 | 0.0080979 |
| FashionMNIST | global | — | 1000 | proposal | 10 | 0.01508 | 0.00071333 | 21.14 | 2.7825e-09 | 0.013772 |
| FashionMNIST | global | — | 1000 | mechanism | 10 | 0.01584 | 0.00062436 | 25.37 | 5.5242e-10 | 0.014695 |
| FashionMNIST | global | — | 1000 | selection | 10 | 0.01623 | 0.00080665 | 20.12 | 4.3058e-09 | 0.014751 |
| FashionMNIST | patch | — | 250 | width3x | 10 | 0.00329 | 0.0010369 | 3.1728 | 0.005657 | 0.0013892 |
| FashionMNIST | patch | — | 250 | proposal | 10 | 0.00728 | 0.00068424 | 10.64 | 1.0654e-06 | 0.0060257 |
| FashionMNIST | patch | — | 1000 | width3x | 10 | 0.00242 | 0.0005004 | 4.8361 | 0.00046299 | 0.0015027 |
| FashionMNIST | patch | — | 1000 | proposal | 10 | 0.00453 | 0.00061555 | 7.3593 | 2.1424e-05 | 0.0034016 |
| CIFAR10 | hypersphere | 4000 | 1000 | obs − random | 10 | 0.01371 | 0.001423 | 9.6344 | 2.4371e-06 | 0.011101 |
| CIFAR10 | receptive_field | 4000 | 1000 | obs − random | 10 | 0.0312 | 0.0014626 | 21.332 | 2.5686e-09 | 0.028519 |
| CIFAR10 | global | — | 250 | width3x | 10 | 0.01857 | 0.001639 | 11.33 | 6.2718e-07 | 0.015565 |
| CIFAR10 | global | — | 250 | proposal | 10 | 0.03068 | 0.0015903 | 19.292 | 6.2374e-09 | 0.027765 |
| CIFAR10 | global | — | 250 | mechanism | 10 | 0.01333 | 0.0015025 | 8.8718 | 4.8001e-06 | 0.010576 |
| CIFAR10 | global | — | 250 | selection | 10 | 0.03409 | 0.0012417 | 27.454 | 2.7374e-10 | 0.031814 |
| CIFAR10 | global | — | 1000 | width3x | 10 | 0.02062 | 0.0012645 | 16.307 | 2.7241e-08 | 0.018302 |
| CIFAR10 | global | — | 1000 | proposal | 10 | 0.02469 | 0.00098075 | 25.175 | 5.9175e-10 | 0.022892 |
| CIFAR10 | global | — | 1000 | mechanism | 10 | 0.0096 | 0.00086999 | 11.035 | 7.8395e-07 | 0.0080052 |
| CIFAR10 | global | — | 1000 | selection | 10 | 0.02446 | 0.0014824 | 16.501 | 2.4576e-08 | 0.021743 |
| CIFAR10 | patch | — | 250 | width3x | 10 | 0.00398 | 0.0017809 | 2.2349 | 0.02614 | 0.00071546 |
| CIFAR10 | patch | — | 250 | proposal | 10 | 0.00576 | 0.0017802 | 3.2356 | 0.0051155 | 0.0024967 |
| CIFAR10 | patch | — | 1000 | width3x | 10 | 0.01032 | 0.0013167 | 7.8377 | 1.3032e-05 | 0.0079063 |
| CIFAR10 | patch | — | 1000 | proposal | 10 | 0.00542 | 0.0011889 | 4.5588 | 0.00068432 | 0.0032406 |
| MNIST (exploratory) | hypersphere | 4000 | 1000 | obs − random | 10 | 0.01229 | 0.00051477 | 23.875 | 9.4746e-10 | 0.011346 |
| MNIST (exploratory) | receptive_field | 4000 | 1000 | obs − random | 10 | 0.01798 | 0.00076432 | 23.524 | 1.0802e-09 | 0.016579 |
| MNIST (exploratory) | global | — | 250 | width3x | 10 | 0.01784 | 0.00044726 | 39.887 | 9.735e-12 | 0.01702 |
| MNIST (exploratory) | global | — | 250 | proposal | 10 | 0.02867 | 0.00044072 | 65.053 | 1.2097e-13 | 0.027862 |
| MNIST (exploratory) | global | — | 250 | mechanism | 10 | 0.00554 | 0.00064742 | 8.557 | 6.4389e-06 | 0.0043532 |
| MNIST (exploratory) | global | — | 250 | selection | 10 | 0.01879 | 0.00083738 | 22.439 | 1.6419e-09 | 0.017255 |
| MNIST (exploratory) | global | — | 1000 | width3x | 10 | 0.00986 | 0.00045709 | 21.571 | 2.3278e-09 | 0.0090221 |
| MNIST (exploratory) | global | — | 1000 | proposal | 10 | 0.01269 | 0.00050254 | 25.252 | 5.7584e-10 | 0.011769 |
| MNIST (exploratory) | global | — | 1000 | mechanism | 10 | 0.00308 | 0.00046899 | 6.5672 | 5.1548e-05 | 0.0022203 |
| MNIST (exploratory) | global | — | 1000 | selection | 10 | 0.0096 | 0.00039553 | 24.271 | 8.1861e-10 | 0.0088749 |
| MNIST (exploratory) | patch | — | 250 | width3x | 10 | 0.01498 | 0.0010086 | 14.852 | 6.147e-08 | 0.013131 |
| MNIST (exploratory) | patch | — | 250 | proposal | 10 | 0.01115 | 0.00080502 | 13.851 | 1.125e-07 | 0.0096743 |
| MNIST (exploratory) | patch | — | 1000 | width3x | 10 | 0.00645 | 0.00030993 | 20.811 | 3.196e-09 | 0.0058819 |
| MNIST (exploratory) | patch | — | 1000 | proposal | 10 | 0.00514 | 0.00037094 | 13.857 | 1.1208e-07 | 0.00446 |

### Subspace convergence

All converged (among recorded subspace solves).

### Diagnostic (no test)

| Dataset | k | Contrast | Δ |
| --- | --- | --- | --- |
| FashionMNIST | 250 | gen-1 val_mse_loop: phd_classshuffle − phd | 0.0010261 ± 7.9891e-05 (n=10) |
| FashionMNIST | 1000 | gen-1 val_mse_loop: phd_classshuffle − phd | 0.0011589 ± 2.8533e-05 (n=10) |
| CIFAR10 | 250 | gen-1 val_mse_loop: phd_classshuffle − phd | 0.00055016 ± 7.1509e-05 (n=10) |
| CIFAR10 | 1000 | gen-1 val_mse_loop: phd_classshuffle − phd | 0.00073774 ± 5.175e-05 (n=10) |
| MNIST (exploratory) | 250 | gen-1 val_mse_loop: phd_classshuffle − phd | -0.00029045 ± 9.9406e-05 (n=10) |
| MNIST (exploratory) | 1000 | gen-1 val_mse_loop: phd_classshuffle − phd | -0.00021564 ± 3.6232e-05 (n=10) |
