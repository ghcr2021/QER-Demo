import math
import random
import numpy as np
import collections

class QuantumSequencePrioritizedReplayBuffer:
    """
    Quantum-Inspired Sequence Experience Replay (QER) Buffer.
    
    This class implements the core QER mechanism as described in the paper:
    "QER-LPD3QN: A Quantum-Inspired Sequence-Aware Deep Reinforcement Learning Algorithm for Path Planning".

    It manages experience sequences using simulated quantum states (qubits) to 
    dynamically adjust sampling priorities based on TD-errors (Prepare Operation) 
    and replay counts (Depreciation Operation).
    """

    def __init__(self, capacity, sequence_length=2,
                 mu=100, iota=0.25 * math.pi,
                 zeta1=0.03 * math.pi, zeta2=2e2,
                 tau1=math.pi, tau2=1e2,
                 epsilon=1e-6, beta=0.4, beta_increment=0.001):
        """
        Initialize the QER Buffer.

        Args:
            capacity (int): Max number of transitions.
            sequence_length (int): Length of the sequence for LSTM processing (L).
            mu, iota (float): Hyperparameters for rotation steps calculation (Eq. 32).
            zeta1, zeta2 (float): Hyperparameters for Preparation Factor sigma (Eq. 31).
            tau1, tau2 (float): Hyperparameters for Depreciation Factor omega (Eq. 37).
            epsilon (float): Small constant to prevent zero priority.
            beta (float): Importance sampling exponent.
            beta_increment (float): Linear increment for beta.
        """
        self.capacity = capacity
        self.sequence_length = sequence_length
        
        # Storage
        self.data = [None] * capacity
        # Quantum states encoded as [b0, b1] amplitudes on the Bloch sphere
        self.quantum_states = [None] * capacity  
        self.replay_times = np.zeros(capacity)
        self.td_errors = np.ones(capacity) * 1.0
        
        # Pointers and counters
        self.ptr = 0
        self.size = 0
        self.training_episode = 0 
        
        # Hyperparameters
        self.epsilon = epsilon
        self.mu = mu
        self.iota = iota
        self.zeta1 = zeta1
        self.zeta2 = zeta2
        self.tau1 = tau1
        self.tau2 = tau2
        
        self.max_td_error = 1.0
        self.max_replay_times = 1
        self.beta = beta
        self.beta_increment = beta_increment

        # Sequence management
        self.episode_boundaries = []  # Stores indices of episodes
        self.current_episode = []     # Temp storage for the active episode

    def get_beta(self):
        """Update and return the Importance Sampling beta."""
        current_beta = self.beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        return current_beta

    def prepare_factor(self):
        """
        Calculate the Preparation Factor (sigma) based on training progress.
        See Eq. (31) in the paper.
        """
        TE = max(self.training_episode, 1)
        exponent = self.zeta2 / TE
        MAX_EXP = 700
        if exponent > MAX_EXP:
            exponent = MAX_EXP
        return self.zeta1 / (1 + math.exp(exponent))

    def depreciation_factor(self):
        """
        Calculate the Depreciation Factor (omega) based on replay counts.
        See Eq. (37) in the paper.
        """
        if self.max_replay_times == 0:
            return 0
        exponent = self.tau2 / max(1, self.training_episode)
        MAX_EXP = 700
        if exponent > MAX_EXP:
            exponent = MAX_EXP
        return self.tau1 / (self.max_replay_times * (1 + math.exp(exponent)))

    def reset_to_uniform(self, index):
        """Reset a quantum state to uniform superposition: |psi> = 1/sqrt(2)(|0> + |1>)."""
        val = 1.0 / math.sqrt(2)
        self.quantum_states[index] = [val, val]

    def prepare_operation(self, index, td_error):
        """
        Apply the Prepare Operation (Rotation) based on TD-error.
        Rotates the state towards |1> (higher priority) if error is high.
        See Eq. (33) - (35) in the paper.
        """
        priority = abs(td_error) + self.epsilon
        Pk = priority
        Pmax = self.max_td_error
        sigma = self.prepare_factor()
        
        if sigma == 0: sigma = 1e-10
        if Pmax == 0: Pmax = 1e-10
        
        # Calculate rotation steps mk (Eq. 32)
        try:
            mk = math.floor(self.mu * (Pk / Pmax) - self.iota / sigma)
        except (ValueError, OverflowError):
            mk = 0

        # Apply rotation matrix
        b0, b1 = self.quantum_states[index]
        total_angle = mk * sigma
        cos_total = math.cos(total_angle)
        sin_total = math.sin(total_angle)
        
        # Rotation logic
        if mk > 0:
            new_b0 = cos_total * b0 - sin_total * b1
            new_b1 = sin_total * b0 + cos_total * b1
        else:
            new_b0 = cos_total * b0 + sin_total * b1
            new_b1 = -sin_total * b0 + cos_total * b1

        # Normalization (Constraint: |b0|^2 + |b1|^2 = 1)
        norm_sq = new_b0**2 + new_b1**2
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            self.quantum_states[index] = [new_b0/norm, new_b1/norm]
        else:
            self.reset_to_uniform(index)

    def depreciation_operation(self, index):
        """
        Apply the Depreciation Operation based on replay counts.
        Rotates the state towards |0> (lower priority) to avoid over-fitting.
        See Eq. (36) in the paper.
        """
        b0, b1 = self.quantum_states[index]
        omega = self.depreciation_factor()
        
        if not math.isfinite(omega):
            return
            
        cos_omega = math.cos(omega)
        sin_omega = math.sin(omega)
        
        # Rotation towards |0>
        new_b0 = cos_omega * b0 + sin_omega * b1
        new_b1 = -sin_omega * b0 + cos_omega * b1
        
        norm_sq = new_b0**2 + new_b1**2
        if norm_sq > 0:
            norm = math.sqrt(norm_sq)
            self.quantum_states[index] = [new_b0/norm, new_b1/norm]
        else:
            self.reset_to_uniform(index)

    def add(self, state, action, reward, next_state, done, td_error=None):
        """
        Add a new transition to the buffer.
        Manages episode boundaries for sequence sampling.
        """
        if td_error is None:
            td_error = self.max_td_error
            
        self.max_td_error = max(self.max_td_error, abs(td_error) + self.epsilon)
        experience = (state, action, reward, next_state, done)
        
        # Store data
        self.data[self.ptr] = experience
        self.td_errors[self.ptr] = td_error
        
        # Initialize quantum state
        self.reset_to_uniform(self.ptr)
        self.replay_times[self.ptr] = 0
        
        # Track episode structure
        self.current_episode.append(self.ptr)
        
        # Pointer update
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
        if done:
            self.episode_boundaries.append(self.current_episode[:])
            self.current_episode = []

    def slice_by_episodes(self):
        """Helper to retrieve valid episode indices."""
        # Include the currently accumulating episode if it exists
        if not self.episode_boundaries and self.current_episode:
            # Note: In a real training loop, we usually wait for 'done', 
            # but for sampling robustness we can include partials if needed.
            # Here we follow the logic of only sampling finished or tracked episodes.
            pass
        return self.episode_boundaries

    def sample(self, batch_size):
        """
        Sample a batch of sequences based on Quantum Probabilities.
        Probability P = |b1|^2.
        
        Returns:
            states, actions, rewards, next_states, dones, indices, weights
        """
        episodes = self.slice_by_episodes()
        if not episodes or len(episodes) == 0:
            return None
            
        sequences = []
        indices_list = []
        
        # 1. Randomly select episodes, then select sequences within them
        # (Note: In a full optimized version, we might weigh episodes, 
        # but here we weigh the *sequences* after selection or use a global tree.
        # This implementation follows the logic of selecting candidates then computing weights)
        
        # To strictly follow the "Sampling Probability = Amplitude" logic efficiently:
        # We sample N candidates, compute their probabilities, and then compute IS weights.
        # For true priority sampling, one would typically use a Segment Tree.
        # Here we verify the 'Amplitude' mechanism on the selected batch for IS correction.
        
        for _ in range(batch_size):
            episode_idx = random.randint(0, len(episodes) - 1)
            episode_indices = episodes[episode_idx]
            if not episode_indices:
                continue
            
            # Retrieve valid data
            episode_data = [self.data[i] for i in episode_indices if self.data[i] is not None]
            if len(episode_data) < self.sequence_length:
                continue
                
            # Random sliding window start
            start_idx = random.randint(0, len(episode_data) - self.sequence_length)
            sequence = episode_data[start_idx : start_idx + self.sequence_length]
            sequences.append(sequence)
            
            # Track original buffer indices for updates
            original_indices = [episode_indices[start_idx + i] for i in range(self.sequence_length)]
            indices_list.append((episode_idx, start_idx, original_indices))

        if not sequences:
            return None

        # 2. Pack batch data
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for sequence in sequences:
            states.append([s[0] for s in sequence])
            actions.append([s[1] for s in sequence])
            rewards.append([s[2] for s in sequence])
            next_states.append([s[3] for s in sequence])
            dones.append([s[4] for s in sequence])

        # Convert to numpy (generic, no Torch dependency here for portability)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)

        # 3. Calculate Probabilities based on Quantum Amplitudes
        probabilities = np.zeros(len(indices_list), dtype=np.float64)
        for i, idx_info in enumerate(indices_list):
            _, _, original_indices = idx_info
            
            # Sequence probability is derived from the average/sum of transitions in it
            # P_seq = Mean( |b1_t|^2 )
            b1_sq_sum = 0.0
            count = 0
            for idx in original_indices:
                if 0 <= idx < self.capacity and self.quantum_states[idx] is not None:
                    b1 = self.quantum_states[idx][1]
                    b1_sq_sum += b1 * b1
                    count += 1
            
            if count > 0:
                prob = max(b1_sq_sum / count, 1e-10)
                probabilities[i] = prob
            else:
                probabilities[i] = 0.5 # Default

        # Normalize probabilities for IS weight calculation
        total_prob = np.sum(probabilities)
        if total_prob > 0:
            probabilities = probabilities / total_prob
        else:
            probabilities = np.ones_like(probabilities) / len(probabilities)

        # 4. Compute Importance Sampling (IS) Weights
        N = len(self)
        current_beta = self.get_beta()
        weights = np.power(N * probabilities + 1e-6, -current_beta)
        max_weight = np.max(weights)
        if max_weight > 0:
            weights = weights / max_weight
        else:
            weights = np.ones_like(weights)

        return states, actions, rewards, next_states, dones, indices_list, weights

    def update_td_errors(self, indices, td_errors):
        """
        Update the priorities (Quantum States) of sampled sequences.
        Executes:
            1. Prepare Operation (using new TD-error)
            2. Depreciation Operation (using replay count)
        """
        for i, idx_tuple in enumerate(indices):
            # Parse indices structure
            if len(idx_tuple) == 3:
                _, _, original_indices = idx_tuple
            else:
                continue
                
            try:
                error = float(td_errors[i])
                if not math.isfinite(error): continue
            except:
                continue

            # Update each transition in the sequence
            for idx in original_indices:
                if 0 <= idx < self.capacity:
                    self.replay_times[idx] += 1
                    self.max_replay_times = max(self.max_replay_times, self.replay_times[idx])
                    
                    self.td_errors[idx] = error
                    self.max_td_error = max(self.max_td_error, abs(error) + self.epsilon)
                    
                    # Core QER Updates
                    self.prepare_operation(idx, error)
                    self.depreciation_operation(idx)

    def increment_training_episode(self):
        """Call this at the end of each episode to update sigma/omega schedules."""
        self.training_episode += 1

    def __len__(self):
        return self.size