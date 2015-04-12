library(MDPtoolbox)

mdp_example_rand(2, 2)

# Generates a MDP for a simple forest management problem
MDP <- mdp_example_forest(S=20, r1=20, r2=2, p=0.9)

# Find an optimal policy via policy iteration
results <- mdp_policy_iteration(MDP$P, MDP$R, 0.9)

# Visualize the policy
results$policy
#(results$iter)

# Find an optimal policy via value iteration
results <- mdp_value_iteration(MDP$P, MDP$R, 0.9)

# Visualize the policy
results$policy
#(results$iter)

# Find an optimal policy via Q-learning
results <- mdp_Q_learning(MDP$P, MDP$R, 0.9, N=10000)

# Visualize the policy
results$policy
#(results$Q)