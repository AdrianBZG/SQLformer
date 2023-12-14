# SQLformer

Official implementation of "SQLformer: Deep Auto-Regressive Query Graph Generation for Text-to-SQL Translation"

# Overview

SQLformer is a novel Transformer architecture specifically crafted to perform text-to-SQL translation tasks. The model predicts SQL queries as abstract syntax trees (ASTs) in an autoregressive way, incorporating structural inductive bias in the encoder and decoder layers. This bias, guided by database table and column selection, aids the decoder in generating SQL query ASTs represented as graphs in a Breadth-First Search canonical order. Comprehensive experiments illustrate the effectiveness of SQLformer on the Spider benchmark.

# Graphical summary of the proposed architecture

![SQLformer](https://i.gyazo.com/e869c8f3e81876b6daa86caad62b19f3.png)

# License

SQLformer is released under the [MIT](LICENSE).

# Note: Repository in progress

We are still polishing up the code and will submit the updated version soon. Thank you for your interest in our work. For any question pleae contact Adrián Bazaga (ar989 (at) cam.ac.uk)

# Citation

`@misc{bazaga2023sqlformer,
      title={SQLformer: Deep Auto-Regressive Query Graph Generation for Text-to-SQL Translation}, 
      author={Adrián Bazaga and Pietro Liò and Gos Micklem},
      year={2023},
      eprint={2310.18376},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}`
