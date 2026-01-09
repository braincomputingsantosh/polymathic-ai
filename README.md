# Chaos Across Domains: Supernovae & Neural Dynamics

Code and visualizations accompanying the Medium article series:
- [Part 1: The Hidden Connection Between Supernovae and Your Brain](https://medium.com/@srinivas.santosh/when-stars-explode-and-neurons-fire-the-hidden-connection-between-supernovae-and-your-brain-febb765d3893)
- Part 2: A Hands-On Guide to Measuring Chaos

## What's This About?

Exploding stars and thinking brains share the same mathematical language — chaos theory, reaction-diffusion dynamics, and Lyapunov exponents. This repo provides Python code to explore these connections.

## Quick Start

```bash
pip install numpy scipy matplotlib
python generate_v2_figures.py
```

This generates 8 figures comparing neural and supernova dynamics.

## Files

| File | Description |
|------|-------------|
| `generate_v2_figures.py` | Main script — generates all article visualizations |
| `lyapunov_well.py` | Lyapunov exponent computation toolkit |
| `cross_domain_analysis.py` | Neural & supernova simulation models |

## Key Concepts

- **Lyapunov Exponents** — Quantify chaos by measuring how fast nearby trajectories diverge
- **FitzHugh-Nagumo Model** — Simplified neural dynamics showing wave propagation
- **Reaction-Diffusion Systems** — The mathematical bridge between domains

## Sample Output

<img width="1745" height="1183" alt="fig5_comparison" src="https://github.com/user-attachments/assets/f55330c2-0a91-46b0-943d-818ae37c0179" />
*Neural waves (top) vs supernova shocks (bottom) — different physics, similar dynamics*

## References

- [The Well Dataset](https://polymathic-ai.org/the_well/) — 15TB of physics simulations from Polymathic AI
- Rosenstein et al. (1993) — Practical Lyapunov exponent computation
- FitzHugh (1961) — Neural membrane models

## License

MIT

---

*Questions? Open an issue or reach out on [[LinkedIn]([https://linkedin.com/in/santoshms](https://www.linkedin.com/in/santoshms/)).*](https://www.linkedin.com/in/santoshms/)
