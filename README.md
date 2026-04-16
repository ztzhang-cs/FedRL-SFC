Activate your Python virtual environment (Windows PowerShell) and install the required dependencies:

(Note: If you are using Linux/macOS, activate the environment using source venv/bin/activate)

2. Running Experiments
The framework is evaluated across two distinct topological scenarios as described in the paper:

Toy Environment: A structured network engineered with severe asymmetric bottlenecks (6 domains).

Random Environment: A complex, procedurally generated multi-domain mesh network (12 domains).

To train and evaluate the FedRL framework, run the main experiment script with your desired mode:

Run the Toy Environment:

Run the Random Environment:

📊 Reproducibility & Pre-trained Models
Due to the inherent stochasticity of Reinforcement Learning (RL) and dynamic network traffic generation, training the framework from scratch without fixed seeds may lead to slight variance in performance.

📝 Citation
If you find this code or our framework useful in your research, please consider citing our paper once it is published:

📄 License
This project is licensed under the MIT License - see the  file for details.
