import numpy as np
import random
import matplotlib.pyplot as plt


# Agent Class
class Agent:
    def __init__(self, name, capability, resource_need, selfishness):
        self.name = name
        self.capability = capability  # How much the agent contributes
        self.resource_need = resource_need  # Resources required by the agent
        self.selfishness = selfishness  # Tendency to prioritize self-interest
        self.payoff = 0  # Accumulated payoff
        self.resources_allocated = 0  # Resources received in the game
        self.strategy = 1.0  # Learning parameter for bidding

    def bid(self, resource_pool):
        """Calculate bid based on resource need, selfishness, and strategy."""
        bid = self.resource_need * (1 + self.selfishness * self.strategy)
        return min(bid, resource_pool)  # Bid cannot exceed available resources

    def update_strategy(self, reward):
        """Adjust strategy based on reward feedback."""
        if reward > 0:
            self.strategy *= 1.1  # Increase strategy if reward is positive
        else:
            self.strategy *= 0.9  # Decrease strategy if reward is negative


# Resource Pool Class
class ResourcePool:
    def __init__(self, total_resources):
        self.total_resources = total_resources
        self.remaining_resources = total_resources

    def replenish(self):
        """Replenish resources for the next round."""
        self.remaining_resources = self.total_resources

    def allocate(self, bids):
        """Allocate resources based on bids."""
        total_bids = sum(bids.values())
        if total_bids <= self.remaining_resources:
            allocation = bids
        else:
            allocation = {agent: (bid / total_bids) * self.remaining_resources for agent, bid in bids.items()}
        self.remaining_resources -= sum(allocation.values())
        return allocation



# Shapley Value Calculation
def compute_shapley_value(contributions, total_contribution):
    """Compute Shapley value for each agent."""
    return {agent: contribution / total_contribution if total_contribution > 0 else 0 for agent, contribution in
            contributions.items()}


# Game Mechanism
def resource_allocation_game(agents, resource_pool, rounds=10):
    payoff_history = {agent.name: [] for agent in agents}
    allocation_history = {agent.name: [] for agent in agents}

    for round_no in range(rounds):
        print(f"\n--- Round {round_no + 1} ---")

        # Replenish resources
        resource_pool.replenish()

        bids = {agent: agent.bid(resource_pool.remaining_resources) for agent in agents}
        allocation = resource_pool.allocate(bids)

        contributions = {}
        for agent, allocated in allocation.items():
            agent.resources_allocated += allocated
            contributions[agent] = agent.capability * allocated

        total_contribution = sum(contributions.values())
        shapley_values = compute_shapley_value(contributions, total_contribution)

        for agent, shapley_value in shapley_values.items():
            reward = shapley_value * resource_pool.total_resources
            agent.payoff += reward
            agent.update_strategy(reward)

            payoff_history[agent.name].append(agent.payoff)
            allocation_history[agent.name].append(allocation[agent])

        for agent in agents:
            print(f"{agent.name}: Allocated {allocation[agent]:.2f}, Payoff {agent.payoff:.2f}")

    return payoff_history, allocation_history



# Visualization
def visualize_results(payoff_history, allocation_history):
    rounds = len(next(iter(payoff_history.values())))

    # Payoff History
    plt.figure(figsize=(12, 5))
    for agent_name, payoffs in payoff_history.items():
        plt.plot(range(1, rounds + 1), payoffs, label=f"{agent_name} Payoff")
    plt.title("Agent Payoff Over Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Payoff")
    plt.legend()
    plt.grid()
    plt.show()

    # Allocation History
    plt.figure(figsize=(12, 5))
    for agent_name, allocations in allocation_history.items():
        plt.plot(range(1, rounds + 1), allocations, label=f"{agent_name} Allocations")
    plt.title("Agent Resource Allocations Over Rounds")
    plt.xlabel("Rounds")
    plt.ylabel("Resources Allocated")
    plt.legend()
    plt.grid()
    plt.show()


# Main Execution
if __name__ == "__main__":
    # Initialize agents
    agents = [
        Agent("Agent A", capability=5, resource_need=10, selfishness=0.5),
        Agent("Agent B", capability=8, resource_need=8, selfishness=0.3),
        Agent("Agent C", capability=4, resource_need=12, selfishness=0.7),
    ]

    # Initialize resource pool
    resource_pool = ResourcePool(total_resources=30)

    # Run the game
    payoff_history, allocation_history = resource_allocation_game(agents, resource_pool, rounds=10)

    # Visualize results
    visualize_results(payoff_history, allocation_history)
