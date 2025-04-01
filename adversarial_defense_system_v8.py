# Filename: adversarial_defense_system_v8.py

import logging
import os
import re
import string
import base64
import binascii
import numpy as np
from typing import Tuple, Optional, List, Dict, Any

import asyncio
import aiohttp
from urllib.parse import unquote_plus

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import pennylane as qml

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from collections import defaultdict

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import KerasClassifier

from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# Ollama LLM Assistant Class
# ==========================
class OllamaAssistant:
    def __init__(self, api_url: str = os.getenv('OLLAMA_API_URL', 'http://localhost:11434'), 
                 model_name: str = os.getenv('MODEL_NAME', 'llama3.2'), 
                 temperature: float = 0.0):
        self.api_url = api_url
        self.model_name = model_name
        self.temperature = temperature

    async def _check_api_availability(self) -> bool:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, asyncio.TimeoutError):
            logger.warning("Ollama API unavailable, using fallback.")
            return False

    async def analyze_input(self, input_text: str) -> str:
        if not await self._check_api_availability():
            return "Fallback: No adversarial patterns detected due to API unavailability."
        prompt = (
            "You are an AI security system. Analyze the following text input to detect any potential "
            "adversarial patterns or attempts at manipulating an AI model through encoded or obfuscated methods. "
            "Provide an explanation if any adversarial techniques are detected.\n\n"
            f"Input Text: {input_text}"
        )
        return await self._make_api_call(prompt)

    async def explain_prediction(self, input_text: str, prediction: str) -> str:
        if not await self._check_api_availability():
            return f"Fallback: Prediction '{prediction}' due to unavailable API."
        prompt = (
            f"Provide a detailed explanation for why the input text was classified as '{prediction}'. "
            f"Explain what patterns or features led to this classification.\n\n"
            f"Input Text: {input_text}"
        )
        return await self._make_api_call(prompt)

    async def generate_adversarial_example(self, input_text: str) -> str:
        if not await self._check_api_availability():
            return f"Fallback: Adversarial example for '{input_text}' not generated."
        prompt = (
            "Generate an adversarial example from the following input text that could fool an AI model.\n\n"
            f"Input Text: {input_text}"
        )
        return await self._make_api_call(prompt, temperature=0.8)

    async def _make_api_call(self, prompt: str, temperature: Optional[float] = None) -> str:
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "options": {"temperature": temperature if temperature is not None else self.temperature},
            "stream": False
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/api/generate", json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    return data.get('response', '').strip()
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"Error during Ollama API call: {e}")
            return f"Error: {str(e)}"

# ==========================
# Input Sanitizer Class
# ==========================
class InputSanitizer:
    def __init__(self, language: str = 'english'):
        try:
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/words')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('stopwords')
            nltk.download('words')
            nltk.download('wordnet')
        self.stopwords = set(stopwords.words(language)) | set(ENGLISH_STOP_WORDS)
        self.words = set(nltk.corpus.words.words())
        self.lemmatizer = WordNetLemmatizer()

    def normalize_text(self, input_text: str) -> str:
        text = re.sub(r'[\u200b-\u200d\uFEFF]', '', input_text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\d+', '', text)
        return text.lower()

    def remove_stopwords(self, input_text: str) -> str:
        tokens = input_text.split()
        filtered_tokens = [self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords]
        return ' '.join(filtered_tokens)

    def detect_and_invert_transformations(self, input_text: str) -> str:
        text = input_text
        transformations_applied = set()
        for _ in range(3):
            prev_text = text
            for method in ['base64', 'url', 'hex', 'caesar']:
                if method not in transformations_applied:
                    text = getattr(self, f"detect_and_reverse_{method}")(text)
                    if text != prev_text:
                        transformations_applied.add(method)
                        break
            if text == prev_text:
                break
        return text

    def detect_and_reverse_base64(self, input_text: str) -> str:
        try:
            padded_text = input_text + '=' * (4 - len(input_text) % 4 if len(input_text) % 4 else 0)
            decoded_bytes = base64.b64decode(padded_text)
            decoded_text = decoded_bytes.decode('utf-8')
            if self.is_printable(decoded_text):
                logger.info("Detected and reversed Base64 encoding.")
                return decoded_text
            return input_text
        except (binascii.Error, UnicodeDecodeError):
            return input_text

    def detect_and_reverse_url(self, input_text: str) -> str:
        decoded_text = unquote_plus(input_text)
        if decoded_text != input_text and self.is_printable(decoded_text):
            logger.info("Detected and reversed URL encoding.")
            return decoded_text
        return input_text

    def detect_and_reverse_hex(self, input_text: str) -> str:
        try:
            decoded_text = bytes.fromhex(input_text).decode('utf-8')
            if self.is_printable(decoded_text):
                logger.info("Detected and reversed hex encoding.")
                return decoded_text
            return input_text
        except ValueError:
            return input_text

    def detect_and_reverse_caesar(self, input_text: str) -> str:
        max_matches = 0
        best_decoded = input_text
        for shift in range(1, 26):
            decoded_text = ''.join(
                chr((ord(char) - ord('a' if char.islower() else 'A') - shift) % 26 + ord('a' if char.islower() else 'A'))
                if char.isalpha() else char
                for char in input_text
            )
            tokens = decoded_text.split()
            matches = sum(1 for word in tokens if word.lower() in self.words)
            if matches > max_matches:
                max_matches = matches
                best_decoded = decoded_text
        if max_matches > 0 and self.is_printable(best_decoded):
            logger.info(f"Detected and reversed Caesar Cipher with {max_matches} matches.")
            return best_decoded
        return input_text

    def is_printable(self, text: str) -> bool:
        return all(char in string.printable for char in text)

    def heuristic_check(self, input_text: str) -> str:
        tokens = input_text.split()
        unique_tokens = set(tokens)
        non_ascii = sum(1 for char in input_text if ord(char) > 127)
        entropy = self.calculate_entropy(tokens)
        if not tokens:
            return "Suspicious: Empty content"
        if len(unique_tokens) < len(tokens) * 0.5:
            return "Suspicious: Too repetitive"
        if len(tokens) > 200:
            return "Suspicious: Unusually long"
        if non_ascii > len(input_text) * 0.1:
            return "Suspicious: Excessive non-ASCII"
        if entropy < 3.0:
            return "Suspicious: Low entropy"
        return "Clean"

    def calculate_entropy(self, tokens: List[str]) -> float:
        from collections import Counter
        counts = Counter(tokens)
        probs = [count / len(tokens) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def sanitize_input(self, input_text: str) -> Tuple[str, str]:
        text = self.normalize_text(input_text)
        text = self.remove_stopwords(text)
        text = self.detect_and_invert_transformations(text)
        heuristic_result = self.heuristic_check(text)
        return text, heuristic_result

# ==========================
# Quantum-Classical Adversarial Perturbation Training
# ==========================
class AdversarialPerturbationTraining:
    def __init__(self, model: Sequential, input_dim: int, n_qubits: int = 4, epsilon_fgsm: float = 0.1, epsilon_pgd: float = 0.1):
        self.model = model
        self.input_dim = input_dim
        self.n_qubits = min(n_qubits, input_dim)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)
        self.art_classifier = KerasClassifier(model=model, clip_values=(0, 1))
        self.epsilon_fgsm = epsilon_fgsm
        self.epsilon_pgd = epsilon_pgd
        self.fgsm = FastGradientMethod(estimator=self.art_classifier, eps=self.epsilon_fgsm)
        self.pgd = ProjectedGradientDescent(estimator=self.art_classifier, eps=self.epsilon_pgd)
        self.adversarial_log: List[Dict[str, Any]] = []
        self.quantum_weights = np.random.uniform(0, np.pi, (2, self.n_qubits))

    @qml.qnode(qml.device("default.qubit", wires=4))
    def quantum_classifier(self, x, weights):
        for i in range(self.n_qubits):
            qml.RX(x[i % len(x)], wires=i)
        for layer in range(2):
            for i in range(self.n_qubits):
                qml.RY(weights[layer, i], wires=i)
            qml.CNOT(wires=[0, 1])
            if self.n_qubits > 2:
                qml.CNOT(wires=[2, 3])
        return qml.expval(qml.PauliZ(0))

    def detect_adversarial_quantum(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            norm_x = x / (np.linalg.norm(x) + 1e-10)
            pred = self.quantum_classifier(norm_x[:self.n_qubits], self.quantum_weights)
            preds.append(1 if pred > 0 else 0)
        return np.array(preds)

    def _validate_input(self, X: np.ndarray):
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {X.shape[1]}")

    def generate_adversarial_samples(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self._validate_input(X)
        logger.info("Generating adversarial samples...")
        fgsm_samples = self.fgsm.generate(X)
        pgd_samples = self.pgd.generate(X)
        self.log_adversarial_samples(fgsm_samples, pgd_samples)
        return fgsm_samples, pgd_samples

    def adversarial_training(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 64):
        fgsm_samples, pgd_samples = self.generate_adversarial_samples(X)
        combined_X = np.vstack([X, fgsm_samples, pgd_samples])
        combined_y = np.hstack([y, y, y])
        quantum_preds = self.detect_adversarial_quantum(combined_X)
        logger.info(f"Quantum adversarial detection: {np.mean(quantum_preds):.2f} adversarial rate")
        history = self.model.fit(combined_X, combined_y, epochs=epochs, batch_size=batch_size, verbose=0)
        self.log_training_results(history)
        logger.info("Adversarial training completed.")

    def log_adversarial_samples(self, fgsm_samples: np.ndarray, pgd_samples: np.ndarray):
        self.adversarial_log.append({
            "fgsm_samples": f"FGSM samples: {len(fgsm_samples)}",
            "pgd_samples": f"PGD samples: {len(pgd_samples)}"
        })
        logger.info(self.adversarial_log[-1]["fgsm_samples"])
        logger.info(self.adversarial_log[-1]["pgd_samples"])

    def log_training_results(self, history):
        loss = history.history['loss'][-1]
        acc = history.history.get('accuracy', [None])[-1]
        self.adversarial_log.append({"loss": loss, "accuracy": acc})
        logger.info(f"Training results: Loss={loss}, Accuracy={acc}")

# ==========================
# GAN-Based Adversarial Defense Class
# ==========================
class GANBasedDefense:
    def __init__(self, input_dim: int = 100, adversarial_output_dim: int = 10, hidden_units: int = 1024):
        self.input_dim = input_dim
        self.adversarial_output_dim = adversarial_output_dim
        self.generator = self.build_generator(hidden_units)
        self.discriminator = self.build_discriminator(hidden_units)
        self.discriminator.compile(optimizer=Adam(learning_rate=0.0002), loss='binary_crossentropy', metrics=['accuracy'])
        self.discriminator.trainable = False
        self.gan = self.build_gan()
        self.adversarial_log: List[Dict[str, Any]] = []

    def build_generator(self, hidden_units: int) -> Sequential:
        model = Sequential([
            Dense(hidden_units // 4, activation='relu', input_dim=self.input_dim),
            Dropout(0.3),
            Dense(hidden_units // 2, activation='relu'),
            Dense(hidden_units, activation='relu'),
            Dense(self.adversarial_output_dim, activation='tanh')
        ])
        return model

    def build_discriminator(self, hidden_units: int) -> Sequential:
        model = Sequential([
            Dense(hidden_units, activation='relu', input_dim=self.adversarial_output_dim),
            Dropout(0.3),
            Dense(hidden_units // 2, activation='relu'),
            Dense(hidden_units // 4, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        return model

    def build_gan(self) -> Sequential:
        model = Sequential([self.generator, self.discriminator])
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy')
        return model

    def train_gan(self, real_data: np.ndarray, labels: np.ndarray, epochs: int = 100, batch_size: int = 64, 
                  validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        for epoch in range(epochs):
            noise = np.random.normal(0, 1, (batch_size, self.input_dim))
            adversarial_samples = self.generator.predict(noise, verbose=0)
            idx = np.random.randint(0, real_data.shape[0], batch_size)
            real_batch = real_data[idx]
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            d_loss_real = self.discriminator.train_on_batch(real_batch, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(adversarial_samples, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            g_loss = self.gan.train_on_batch(noise, real_labels)
            self.log_adversarial_training(epoch, d_loss[0], d_loss[1], g_loss)
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}: D Loss={d_loss[0]}, D Acc={d_loss[1]}, G Loss={g_loss}")
            if validation_data:
                val_loss, val_acc = self.validate_gan(validation_data)
                logger.info(f"Validation: Loss={val_loss}, Acc={val_acc}")
        logger.info("GAN training completed.")

    def log_adversarial_training(self, epoch: int, d_loss: float, d_acc: float, g_loss: float):
        self.adversarial_log.append({
            'epoch': epoch + 1,
            'discriminator_loss': d_loss,
            'discriminator_accuracy': d_acc,
            'generator_loss': g_loss
        })

    def validate_gan(self, validation_data: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, float]:
        val_data, val_labels = validation_data
        return self.discriminator.evaluate(val_data, val_labels, verbose=0)

# ==========================
# Differential Privacy Trainer
# ==========================
class DifferentialPrivacyTrainer:
    def __init__(self, model: Sequential, l2_norm_clip: float = float(os.getenv('L2_NORM_CLIP', 1.0)),
                 noise_multiplier: float = float(os.getenv('NOISE_MULTIPLIER', 1.1)),
                 learning_rate: float = float(os.getenv('LEARNING_RATE', 0.01))):
        self.model = model
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        self.learning_rate = learning_rate
        self.optimizer = DPKerasSGDOptimizer(
            l2_norm_clip=self.l2_norm_clip,
            noise_multiplier=self.noise_multiplier,
            learning_rate=self.learning_rate
        )
        self.privacy_budget_log: List[float] = []

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 10, batch_size: int = 64, 
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None):
        logger.info("Starting Differential Privacy training...")
        self.X_train_size = X_train.shape[0]
        self.model.compile(optimizer=self.optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data, verbose=0)
        self.log_privacy_budget(epochs, batch_size)
        logger.info(f"Differential Privacy training completed for {epochs} epochs.")

    def log_privacy_budget(self, epochs: int, batch_size: int):
        delta = 1e-5
        try:
            epsilon = compute_dp_sgd_privacy.compute_dp_sgd_privacy(
                n=self.X_train_size,
                batch_size=batch_size,
                noise_multiplier=self.noise_multiplier,
                epochs=epochs,
                delta=delta
            )[0]
            self.privacy_budget_log.append(epsilon)
            logger.info(f"Privacy budget (ε): {epsilon:.4f}")
        except Exception as e:
            logger.error(f"Error computing privacy budget: {e}")

# ==========================
# Quantum-Enhanced Federated Attack Database
# ==========================
class FederatedAttackDatabase:
    def __init__(self, n_qubits: int = 4):
        self.attack_database: Dict[str, List[np.ndarray]] = {}
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(qml.device("default.qubit", wires=4))
    def quantum_kernel(self, x1, x2):
        for i in range(self.n_qubits):
            qml.RY(x1[i % len(x1)], wires=i)
            qml.RY(x2[i % len(x2)], wires=i)
        return qml.probs(wires=range(self.n_qubits))

    def record_attack(self, system_id: str, attack_signature: np.ndarray):
        if system_id not in self.attack_database:
            self.attack_database[system_id] = []
        self.attack_database[system_id].append(attack_signature)
        logger.info(f"Attack recorded for system {system_id}.")

    def detect_similar_attack(self, attack_signature: np.ndarray, similarity_metric: str = "quantum") -> Optional[str]:
        for system_id, signatures in self.attack_database.items():
            for sig in signatures:
                if self.similarity_check(attack_signature, sig, similarity_metric):
                    logger.info(f"Similar attack in {system_id}.")
                    return system_id
        return None

    def similarity_check(self, sig1: np.ndarray, sig2: np.ndarray, metric: str = "quantum") -> bool:
        sig1_flat = sig1.flatten()
        sig2_flat = sig2.flatten()
        if metric == "quantum":
            norm1 = sig1_flat / (np.linalg.norm(sig1_flat) + 1e-10)
            norm2 = sig2_flat / (np.linalg.norm(sig2_flat) + 1e-10)
            probs = self.quantum_kernel(norm1[:self.n_qubits], norm2[:self.n_qubits])
            return probs[0] > 0.85
        elif metric == "cosine":
            return cosine_similarity(sig1_flat.reshape(1, -1), sig2_flat.reshape(1, -1))[0][0] > 0.9
        elif metric == "euclidean":
            return np.linalg.norm(sig1_flat - sig2_flat) < 0.1
        raise ValueError(f"Unknown metric: {metric}")

# ==========================
# Early Warning System Class
# ==========================
class EarlyWarningSystem:
    def __init__(self, suspicious_threshold: int = int(os.getenv('SUSPICIOUS_THRESHOLD', 5)),
                 long_input_threshold: int = int(os.getenv('LONG_INPUT_THRESHOLD', 200)),
                 max_retraining_freq: int = 10):
        self.suspicious_patterns = 0
        self.suspicious_threshold = suspicious_threshold
        self.long_input_threshold = long_input_threshold
        self.repetitiveness_threshold = 0.5
        self.blocked_inputs: List[str] = []
        self.warning_log: List[Dict[str, Any]] = []
        self.retraining_frequency = 1
        self.max_retraining_freq = max_retraining_freq
        self.retraining_schedule: List[Dict[str, Any]] = []
        self.federated_learning_system = FederatedAttackDatabase()

    def track_and_warn(self, input_text: str):
        tokens = input_text.split()
        unique_tokens = set(tokens)
        non_ascii = sum(1 for char in input_text if ord(char) > 127)
        entropy = self.calculate_entropy(tokens)
        triggers = {
            "repetitive": len(unique_tokens) < len(tokens) * self.repetitiveness_threshold,
            "long": len(tokens) > self.long_input_threshold,
            "non_ascii": non_ascii > len(input_text) * 0.1,
            "low_entropy": entropy < 3.0
        }
        for key, triggered in triggers.items():
            if triggered:
                self.suspicious_patterns += 1
                msg = f"Warning: {key.replace('_', ' ').capitalize()} detected."
                logger.warning(msg)
                self.log_warning(input_text, msg)
        if self.suspicious_patterns >= self.suspicious_threshold:
            self.trigger_enhanced_defense(input_text)

    def calculate_entropy(self, tokens: List[str]) -> float:
        from collections import Counter
        counts = Counter(tokens)
        probs = [count / len(tokens) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)

    def trigger_enhanced_defense(self, input_text: str):
        logger.info("Activating enhanced defenses...")
        self.block_suspicious_input(input_text)
        self.increase_retraining_frequency()
        self.escalate_to_federated_learning(input_text)
        self.suspicious_patterns = 0

    def block_suspicious_input(self, input_text: str):
        self.blocked_inputs.append(input_text)
        logger.info(f"Blocked: {input_text}")

    def log_warning(self, input_text: str, warning_msg: str):
        self.warning_log.append({"input": input_text, "warning": warning_msg})

    def increase_retraining_frequency(self):
        if self.retraining_frequency < self.max_retraining_freq:
            self.retraining_frequency += 1
            self.retraining_schedule.append({
                "frequency": self.retraining_frequency,
                "action": "Increased due to suspicious activity"
            })
            logger.info(f"Retraining frequency: {self.retraining_frequency}")

    def escalate_to_federated_learning(self, input_text: str):
        signature = np.frombuffer(input_text.encode('utf-8'), dtype=np.uint8)
        self.federated_learning_system.record_attack("Current_System", signature)

# ==========================
# Adversarial Defense System Class with Quantum Boost
# ==========================
class AdversarialDefenseSystem:
    def __init__(self, api_url: str = os.getenv('OLLAMA_API_URL', 'http://localhost:11434'),
                 model_name: str = os.getenv('MODEL_NAME', 'llama3.2'),
                 input_dim: int = 10, n_qubits: int = 4):
        self.sanitizer = InputSanitizer()
        perturbation_model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dense(64, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        perturbation_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.detector = AdversarialPerturbationTraining(perturbation_model, input_dim, n_qubits)
        self.gan_defense = GANBasedDefense(input_dim=100, adversarial_output_dim=input_dim)
        self.llm_assistant = OllamaAssistant(api_url=api_url, model_name=model_name)
        self.early_warning = EarlyWarningSystem()
        self.federated_database = FederatedAttackDatabase(n_qubits=n_qubits)
        self.dp_trainer: Optional[DifferentialPrivacyTrainer] = None
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.quantum_weights = np.random.uniform(0, np.pi, (2, n_qubits))

    @qml.qnode(qml.device("default.qubit", wires=4))
    def quantum_defense_classifier(self, x, weights):
        for i in range(self.n_qubits):
            qml.RZ(x[i % len(x)], wires=i)
        for layer in range(2):
            for i in range(self.n_qubits):
                qml.RX(weights[layer, i], wires=i)
            qml.CZ(wires=[0, 1])
            if self.n_qubits > 2:
                qml.CZ(wires=[2, 3])
        return qml.expval(qml.PauliY(0))

    def classify_quantum(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for x in X:
            norm_x = x / (np.linalg.norm(x) + 1e-10)
            pred = self.quantum_defense_classifier(norm_x[:self.n_qubits], self.quantum_weights)
            preds.append(1 if pred > 0 else 0)
        return np.array(preds)

    async def sanitize_and_detect(self, input_text: str, input_vector: np.ndarray, lightweight: bool = False):
        if input_vector.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input dim {self.input_dim}, got {input_vector.shape[-1]}")

        sanitized_text, heuristic_result = self.sanitizer.sanitize_input(input_text)
        logger.info(f"Sanitized: {sanitized_text}")
        logger.info(f"Heuristic: {heuristic_result}")

        ollama_analysis = await self.llm_assistant.analyze_input(sanitized_text)
        logger.info(f"Ollama: {ollama_analysis}")

        self.early_warning.track_and_warn(sanitized_text)

        input_vector = input_vector.reshape(1, -1)
        y_dummy = np.array([1])

        # Quantum classification
        quantum_pred = self.classify_quantum(input_vector)
        logger.info(f"Quantum defense prediction: {'Adversarial' if quantum_pred[0] else 'Clean'}")

        self.detector.adversarial_training(input_vector, y_dummy, epochs=1)

        if not lightweight:
            self.gan_defense.train_gan(input_vector, y_dummy, epochs=1)
            if self.dp_trainer is None:
                model = Sequential([
                    Dense(128, activation='relu', input_dim=self.input_dim),
                    Dense(64, activation='relu'),
                    Dense(1, activation='sigmoid')
                ])
                self.dp_trainer = DifferentialPrivacyTrainer(model)
            X_train = np.random.rand(100, self.input_dim)
            y_train = np.random.randint(0, 2, 100)
            self.dp_trainer.train(X_train, y_train)

        self.federated_database.record_attack("System_1", input_vector.flatten())
        similar = self.federated_database.detect_similar_attack(input_vector.flatten(), "quantum")
        if similar:
            logger.info(f"Quantum-detected similar attack from {similar}")

# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    async def main():
        defense_system = AdversarialDefenseSystem()
        test_cases = [
            ("VGVzdCBpcyBnb29kIQ==", np.random.rand(10)),  # Base64: "Test is good!"
            ("A" * 300, np.random.rand(10)),  # Long input
            ("test " * 10, np.random.rand(10))  # Repetitive
        ]
        for text, vector in test_cases:
            logger.info(f"\nTesting: {text[:50]}...")
            await defense_system.sanitize_and_detect(text, vector, lightweight=True)

    asyncio.run(main())