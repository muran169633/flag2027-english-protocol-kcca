# FLAG 2027 face–voice inference

The inference package supports organizer replay of CodaBench submission **1054878**, which recorded a mean Evaluation EER of **9.87%**. It contains 15 frozen matching heads and 15 sets of Gaussian scoring parameters.

Run the supplied scorer from the eight authorized Evaluation inputs listed in the resource inventory. The local replay reproduced all 2,326 scores across the four Evaluation conditions. [Replay instructions](reference/README.md) · [Resource inventory](reference/resources.json) · [Replay comparison](reference/reference/replay-comparison.json)

The procedure is disclosed for compliance review. 

The separate local English KCCA experiment recorded a five-fold development EER of **21.910%**. Its training code and reproduction instructions are documented in the [KCCA experiment archive](kcca-reference/README.md).
