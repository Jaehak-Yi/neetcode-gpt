class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places   
        answer = init   
        for i in range(iterations):
            answer = answer - 2*answer*learning_rate
        return round(answer, 5)
        
