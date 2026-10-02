Embedder: `hashing`

| retriever | self-query | questions | n | recall@5 | MRR@10 |
|---|---|---|---|---|---|
| bm25 | no | filtered | 96 | 0.140 | 0.105 |
| bm25 | no | semantic | 96 | 0.009 | 0.009 |
| bm25 | no | all | 192 | 0.074 | 0.057 |
| bm25 | yes | filtered | 96 | 0.998 | 1.000 |
| bm25 | yes | semantic | 96 | 0.644 | 0.380 |
| bm25 | yes | all | 192 | 0.821 | 0.690 |
| dense | no | filtered | 96 | 0.292 | 0.266 |
| dense | no | semantic | 96 | 0.247 | 0.291 |
| dense | no | all | 192 | 0.270 | 0.279 |
| dense | yes | filtered | 96 | 0.998 | 1.000 |
| dense | yes | semantic | 96 | 0.690 | 0.640 |
| dense | yes | all | 192 | 0.844 | 0.820 |
| hybrid | no | filtered | 96 | 0.286 | 0.246 |
| hybrid | no | semantic | 96 | 0.094 | 0.086 |
| hybrid | no | all | 192 | 0.190 | 0.166 |
| hybrid | yes | filtered | 96 | 0.998 | 1.000 |
| hybrid | yes | semantic | 96 | 0.675 | 0.541 |
| hybrid | yes | all | 192 | 0.837 | 0.770 |
