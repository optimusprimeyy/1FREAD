# FREAD
Sihan Wang, **Zhong Yuan***, Chuan Luo, Hongmei Chen, and Dezhong Peng, [Exploiting fuzzy rough entropy to detect anomalies](FREAD/Paper/2023-FREAD.pdf), International Journal of Approximate Reasoning, In Press, Journal Pre-proof, 20 Nov 2023, DOI: [10.1016/j.ijar.2023.109087](https://doi.org/10.1016/j.ijar.2023.109087). (Code)

## Abstract
Anomaly detection has been used in a wide range of fields. However, most of the current detection methods are only applicable to certain data, ignoring uncertain information such as fuzziness in the data. Fuzzy rough set theory, as an essential mathematical model for granular computing, provides an effective method for processing uncertain data such as fuzziness. Fuzzy rough entropy has been proposed in fuzzy rough set theory and has been employed successfully in data analysis tasks such as feature selection. However, it mainly uses the intersection operation, which may not effectively reflect the similarity between high-dimensional objects. In response to the two challenges mentioned above, distance-based fuzzy rough entropy and its correlation measures are proposed. Further, the proposed fuzzy rough entropy is used to construct the anomaly detection model and the Fuzzy Rough Entropy-based Anomaly Detection (FREAD) algorithm is designed. Finally, the FREAD algorithm is compared and analyzed with some mainstream anomaly detection algorithms (including COF, DIS, INFLO, LDOF, LoOP, MIX, ODIN, SRO, and VarE algorithms) on some publicly available datasets. Experimental results indicate that the FREAD algorithm significantly outperforms other algorithms in terms of performance and flexibility. \textcolor{blue}{The code is publicly available online at [https://github.com/optimusprimeyy/FREAD](https://github.com/optimusprimeyy/FREAD).

## Usage
You can run FREAD.py:
```
clc;
clear all;
format short

load Example.mat

Dataori=Example;

trandata=Dataori;
trandata=normalize(trandata,'range');

sigma=0.6;
anomaly_score=FGAS(trandata,sigma)

```
You can get outputs as follows:
```
anomaly_score =
    0.1390
    0.0539
    0.3785
    0.0011
    0.3655
         0
    0.4530
    1.0000
    0.13
