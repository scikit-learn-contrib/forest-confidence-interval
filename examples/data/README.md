# Auto MPG dataset

`auto_mpg.csv` is a copy of the Auto MPG dataset created by R. Quinlan
(1993), obtained from [OpenML dataset 196](https://www.openml.org/d/196).
The canonical UCI Machine Learning Repository record is:

> Quinlan, R. (1993). *Auto MPG* [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5859H

## Copyright and license

The dataset is licensed under the
[Creative Commons Attribution 4.0 International license (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
The SPDX license identifier is `CC-BY-4.0`. The UCI record does not provide a
separate copyright notice; copyright remains with the original rights
holder(s).

The OpenML representation omits the original car-name identifier. For inclusion
here, the ARFF data were converted to CSV, and the `model` and `class` column
names were changed to `model_year` and `mpg`, respectively. Missing horsepower
values are represented by empty CSV fields.
