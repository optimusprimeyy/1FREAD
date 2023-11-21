# FREAD
Sihan Wang, **Zhong Yuan***, Chuan Luo, Hongmei Chen, and Dezhong Peng, [Exploiting fuzzy rough entropy to detect anomalies](Paper/2023-FREAD.pdf), International Journal of Approximate Reasoning, In Press, Journal Pre-proof, 20 Nov 2023, DOI: [10.1016/j.ijar.2023.109087](https://doi.org/10.1016/j.ijar.2023.109087). (Code)

## Abstract
Anomaly detection has been used in a wide range of fields. However, most of the current detection methods are only applicable to certain data, ignoring uncertain information such as fuzziness in the data. Fuzzy rough set theory, as an essential mathematical model for granular computing, provides an effective method for processing uncertain data such as fuzziness. Fuzzy rough entropy has been proposed in fuzzy rough set theory and has been employed successfully in data analysis tasks such as feature selection. However, it mainly uses the intersection operation, which may not effectively reflect the similarity between high-dimensional objects. In response to the two challenges mentioned above, distance-based fuzzy rough entropy and its correlation measures are proposed. Further, the proposed fuzzy rough entropy is used to construct the anomaly detection model and the Fuzzy Rough Entropy-based Anomaly Detection (FREAD) algorithm is designed. Finally, the FREAD algorithm is compared and analyzed with some mainstream anomaly detection algorithms (including COF, DIS, INFLO, LDOF, LoOP, MIX, ODIN, SRO, and VarE algorithms) on some publicly available datasets. Experimental results indicate that the FREAD algorithm significantly outperforms other algorithms in terms of performance and flexibility. The code is publicly available online at [https://github.com/optimusprimeyy/FREAD](https://github.com/optimusprimeyy/FREAD).

## Usage
You can run FREAD.py:
```
if __name__ == "__main__":
    load_data = loadmat('FREAD_Example.mat')
    trandata = load_data['trandata']

    scaler = MinMaxScaler()
    trandata[:, 1:] = scaler.fit_transform(trandata[:, 1:])

    delta = 0.5
    out_scores = FREAD(trandata, delta)

    print(out_scores)
```
You can get outputs as follows:
```
out_scores =
    0.8621
    0.9448
    0.8373
    0.8630
    0.8090
    0.8945
```

## Citation
If you find FREAD useful in your research, please consider citing:
```
@article{WANG2023109087,
    title = {Exploiting fuzzy rough entropy to detect anomalies},
    author = {Wang, Si Han and Yuan, Zhong and Luo,Chuan and Chen, Hong Mei and Peng, De Zhong},
    journal = {International Journal of Approximate Reasoning},
    pages = {109087},
    year = {2023},
    doi = {10.1016/j.ijar.2023.109087},
    publisher={Elsevier}
}
```
## Contact
If you have any question, please contact [wangsihan0713@foxmail.com](wangsihan0713@foxmail.com).

