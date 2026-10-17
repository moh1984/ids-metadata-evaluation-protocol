# Dataset

`cybersecurity_dataset.csv` — 10,000 network flow records, 13 raw attributes.

## Source and licence

`cybersecurity_dataset.csv` is redistributed, unmodified, from the Kaggle dataset
**Cybersecurity Threat Detection Dataset** by Dhrubang Talukdar, published under
the **MIT License**:

<https://www.kaggle.com/datasets/dhrubangtalukdar/cybersecurity-threat-detection-dataset>

It is included in this repository so that `run_all.sh` reproduces every reported
number without a separate download. Kaggle lists the dataset under the MIT
License, and the file is redistributed here unmodified with the attribution
above. The dataset is the publisher's work, not ours. Users who need a legal
determination for their own redistribution should consult the source page and
their own counsel rather than rely on this note.

## Provenance

The publisher's description does not state how the records were produced. We
infer a **simulated** environment from the file itself: all 10,000 source
addresses and all 10,000 destination addresses are distinct, which is highly
atypical of captured enterprise traffic and is consistent with synthetic
generation, and the `user_agent` field takes only thirteen values.
`src/provenance_checks.py` measures both.

Simulation gives clean labels and controlled attack proportions, which is what
makes a controlled model comparison possible, but the statistical regularities
the classifiers exploit are ultimately those the generator produced. Results
obtained on this data should not be read as estimates of accuracy on production
traffic. See Section 6.1 of the paper, and the note on label conditioning in the
top-level README.

## Schema

| Column | Description | Type |
|---|---|---|
| `timestamp` | Date and time of the flow | Temporal |
| `src_ip` / `dst_ip` | IPv4 addresses (excluded from features) | Categorical |
| `src_port` / `dst_port` | Layer-4 ports | Numerical |
| `protocol` | TCP, UDP or ICMP | Categorical |
| `bytes_sent` | Bytes source → destination | Numerical |
| `bytes_received` | Bytes destination → source | Numerical |
| `user_agent` | HTTP user-agent string | Textual |
| `url` | URL endpoint (null for non-HTTP flows) | Textual |
| `is_internal_traffic` | Internal-network flag | Boolean |
| `label` | 0 = benign, 1 = attack | Binary |
| `attack_type` | Category name, or `benign` | Categorical |

Source IP and destination IP are present in the file but deliberately **excluded**
from the feature pipeline: they carry no signal that generalises across network
environments.

## Class distribution

| Class | Records | Share |
|---|---|---|
| benign | 9,600 | 96.00% |
| brute-force | 108 | 1.08% |
| port-scan | 81 | 0.81% |
| sql-injection | 64 | 0.64% |
| xss | 34 | 0.34% |
| credential-stuffing | 28 | 0.28% |
| ddos | 27 | 0.27% |
| command-injection | 26 | 0.26% |
| exploit-attempt | 22 | 0.22% |
| c2 | 10 | 0.10% |

Nine attack categories plus benign — ten classes, 24:1 benign-to-attack ratio.

Other characteristics: TCP 7,765 / UDP 1,910 / ICMP 325; internal traffic 1,534
(15.34%); `url` is null for 3,232 records.

## Note on the rare categories

C2 contributes 10 records in total, which becomes 7 in training and 3 in test
under a 70/30 stratified split. Per-class metrics for categories at this scale
cannot support inference and are reported for completeness only.
