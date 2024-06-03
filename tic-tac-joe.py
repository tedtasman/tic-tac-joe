import board

'''
for each episode do
    Initialize game state
    while game is not over do
        Get current state
        Use model to get probability distribution over actions
        Sample action from distribution
        Take action, get reward and new state
        Store state, action, reward
    end while
    Update model parameters using stored states, actions, rewards
end for
'''

