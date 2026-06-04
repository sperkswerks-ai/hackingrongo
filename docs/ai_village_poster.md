# RedRongo: An Autonomous Red Team Agent for Undeciphered Cipher Attack

## The Target

Rongorongo — Easter Island's undeciphered script. 15,273 tokens. Unknown substitution
cipher. No key, no bilingual text, no living readers.

## The Agent

RedRongo is an autonomous agent that attacks the cipher using eight adversarial tools:
reconnaissance, known-plaintext exploitation, crib dragging, oracle probing,
supply chain injection (Indus Valley cross-script priors), history query (MLflow),
MCMC chain execution, and hypothesis declaration.

## Why This Is AI Security Research

The agent treats the LM scorer as a black-box oracle and learns its decision surface
through systematic probing — identical in structure to model extraction attacks.
The supply chain injection module raises a novel question: what happens when an
agentic decipherment system ingests external phonetic priors from a second undeciphered
script? This is contaminated training data / supply chain poisoning applied to
historical cryptanalysis.

## Key Results

- p_good = 0.0003 (first quantum hardness certificate for any undeciphered script)
- 72x theoretical Grover speedup; guided MCMC beats Grover by 108x
- Statistically significant visual similarity between Rongorongo and Indus Valley
  script (KS p < 0.01 vs Linear B control)
- Self-training bootstrapped 9 anchors across 2 iterations (+134 bits)
