# Adaptive Strategy Learning in Liar’s Dice Using Multi-Agent Systems

## Project Overview

### Motivation
Liar’s Dice is a classic imperfect-information, multi-agent game that requires probabilistic reasoning, belief modeling, strategic deception, and adaptation to opponents. Each player must reason not only about the environment (dice outcomes), but also about other agents’ hidden information, strategies, and behavioral tendencies. This makes it a natural and well-motivated testbed for studying multi-agent decision making under uncertainty.

The core motivation of this project is to understand how different forms of strategic reasoning emerge in Liar’s Dice, and how agents can exploit patterns in opponents’ behavior to gain an advantage. Beyond simply finding a strong static strategy, the project focuses on **adaptive play**: learning who you are playing against and responding optimally over time.

### Relation to Course Topics
This project directly relates to several central course topics:
*   Multi-agent games with imperfect information
*   Strategic reasoning and equilibrium concepts
*   Opponent modeling and adaptive agents (similar to IOcaine Powder)
*   Empirical evaluation of agent strategies through simulation
*   Reinforcement learning in multi-agent environments

This project aims to combine the theoretical background from the course with RL implementation.

## Research Questions and Techniques

The project will focus on the following concrete research questions:

### 1. What baseline strategies perform well in Liar’s Dice?
We will evaluate performance under different game configurations (number of dice per player, number of players, allowed bid space, wild dice rules, etc.).
*   **Implementation:** We will implement and evaluate several baseline agents, such as:
    *   Random or naive bidding agents.
    *   Rule-based heuristic agents.
    *   Simple probabilistic agents that estimate dice distributions.
    *   Simplified MDP Agents.

### 2. Can reinforcement learning agents discover stronger strategies than hand-crafted baselines?
We will train reinforcement learning agents using self-play and mixed-opponent play.
*   **Techniques:**
    *   Tabular or approximate Q-learning in simplified state spaces.
    *   Policy gradient or actor-critic methods for richer representations.
    *   Reward shaping to encourage consistent strategic behavior.
*   **Scope:** Due to the complexity of the full game, we will explicitly consider simplified versions of Liar’s Dice (e.g., fewer dice or limited bid ranges) to keep learning tractable.

### 3. Can an adaptive agent identify an opponent’s strategy and exploit it in real time?
The core contribution of the project is an agent that is “better than you” in the sense that it observes an opponent’s actions over repeated rounds, classifies the opponent into a behavioral or strategic type, and adjusts its policy to maximize performance against that specific opponent.
*   **Techniques:** Lightweight opponent modeling, strategy clustering, or inference over latent player types inspired by ideas from game theory and inverse optimization.

## Related Work

*   **Equilibrium Approaches:** A representative line of work treats Liar’s Dice as an imperfect-information game and focuses on computing strong, approximately optimal strategies (like Nash equilibrium) without adapting to a specific opponent. Approaches based on Counterfactual Regret Minimization (CFR) aim to be robust and minimally exploitable.
*   **Opponent Modeling:** A complementary line of research focuses on inferring an opponent’s strategy or type from observed actions. Representative work shows that explicitly modeling opponents can outperform equilibrium strategies when facing non-optimal or systematic opponents.// filepath: c:\Users\yonat\CodeProjects\Multi-Agent-AI\project\README.md
# Adaptive Strategy Learning in Liar’s Dice Using Multi-Agent Systems

**Authors:**
*   Shahar Oded (208388918)
*   Iftach Shoham (207613589)

## Project Overview

### Motivation
Liar’s Dice is a classic imperfect-information, multi-agent game that requires probabilistic reasoning, belief modeling, strategic deception, and adaptation to opponents. Each player must reason not only about the environment (dice outcomes), but also about other agents’ hidden information, strategies, and behavioral tendencies. This makes it a natural and well-motivated testbed for studying multi-agent decision making under uncertainty.

The core motivation of this project is to understand how different forms of strategic reasoning emerge in Liar’s Dice, and how agents can exploit patterns in opponents’ behavior to gain an advantage. Beyond simply finding a strong static strategy, the project focuses on **adaptive play**: learning who you are playing against and responding optimally over time.

### Relation to Course Topics
This project directly relates to several central course topics:
*   Multi-agent games with imperfect information
*   Strategic reasoning and equilibrium concepts
*   Opponent modeling and adaptive agents (similar to IOcaine Powder)
*   Empirical evaluation of agent strategies through simulation
*   Reinforcement learning in multi-agent environments

This project aims to combine the theoretical background from the course with RL implementation.

## Research Questions and Techniques

The project will focus on the following concrete research questions:

### 1. What baseline strategies perform well in Liar’s Dice?
We will evaluate performance under different game configurations (number of dice per player, number of players, allowed bid space, wild dice rules, etc.).
*   **Implementation:** We will implement and evaluate several baseline agents, such as:
    *   Random or naive bidding agents.
    *   Rule-based heuristic agents.
    *   Simple probabilistic agents that estimate dice distributions.
    *   Simplified MDP Agents.

### 2. Can reinforcement learning agents discover stronger strategies than hand-crafted baselines?
We will train reinforcement learning agents using self-play and mixed-opponent play.
*   **Techniques:**
    *   Tabular or approximate Q-learning in simplified state spaces.
    *   Policy gradient or actor-critic methods for richer representations.
    *   Reward shaping to encourage consistent strategic behavior.
*   **Scope:** Due to the complexity of the full game, we will explicitly consider simplified versions of Liar’s Dice (e.g., fewer dice or limited bid ranges) to keep learning tractable.

### 3. Can an adaptive agent identify an opponent’s strategy and exploit it in real time?
The core contribution of the project is an agent that is “better than you” in the sense that it observes an opponent’s actions over repeated rounds, classifies the opponent into a behavioral or strategic type, and adjusts its policy to maximize performance against that specific opponent.
*   **Techniques:** Lightweight opponent modeling, strategy clustering, or inference over latent player types inspired by ideas from game theory and inverse optimization.

## Related Work

*   **Equilibrium Approaches:** A representative line of work treats Liar’s Dice as an imperfect-information game and focuses on computing strong, approximately optimal strategies (like Nash equilibrium) without adapting to a specific opponent. Approaches based on Counterfactual Regret Minimization (CFR) aim to be robust and minimally exploitable.
*   **Opponent Modeling:** A complementary line of research focuses on inferring an opponent’s strategy or type from observed actions. Representative work shows that explicitly modeling opponents can outperform equilibrium strategies when facing non-optimal or systematic opponents.

Our focus is on understanding when and why adaptive play outperforms optimal but non-adaptive strategies, using Liar’s Dice as a clean and interpretable testbed.

