
## Neuron selection (2d43012f8b9d7644efb9f1267c17bb6c65043f21; 1340 unique current rows)


### Pruning

| Dataset | Strategy | P | k | Criterion | Test | Δ random | z |
| --- | --- | --- | --- | --- | --- | --- | --- |
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
| MNIST (exploratory) | 250 | gen-1 val_mse_loop: phd_classshuffle − phd | -0.00029045 ± 9.9406e-05 (n=10) |
| MNIST (exploratory) | 1000 | gen-1 val_mse_loop: phd_classshuffle − phd | -0.00021564 ± 3.6232e-05 (n=10) |
