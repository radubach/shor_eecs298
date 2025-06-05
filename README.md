# shor_eecs298

# Shor's Algorithm – Group Project

This repository is for our group project on **Shor’s Algorithm** for our quantum computing class EECS298 Spring 2025.

---

## 🧠 Project Overview

Shor’s algorithm is a quantum algorithm that efficiently factors large integers—something classical algorithms struggle with. Our goals include:

- Understanding the math behind Shor's algorithm
- Implementing it using Python and Qiskit (or another framework)
- Demonstrating it on small numbers (e.g., factoring 15 into 3 × 5)
- Explaining its implications for cryptography

---

## Docs

### 🔹 [📊 Google Slides – Presentation](https://docs.google.com/presentation/d/1VNeY83Y_1_JY5Fi1vOkatUCXGv3TuXXlTg1dWm2p758/edit?usp=sharing)

- Presented in class
- Walks through algorithm and results

### 🔹 [📝 PDF Write-up – Project Report (UCI Access Only)](https://drive.google.com/file/d/1ry9wVSpGH0Zo0eLWq7UjSsTx3Vzv8yyF/view?usp=sharing)

- Background on cryptography and RSA-2048
- motivation for Shor's
- explanation of algorithm
- (Only accessible with UC Irvine account)

---

## 🧪 Code Structure

<pre>
/src # Helper functions for modular exponentiation, QFT, noise models, etc.
/notebooks/initial # Local development notebooks for prototyping
/notebooks/colab # Final Colab notebooks optimized for cloud execution
</pre>

---

## 💻 Google Colab Notebooks

Click the badge to open each notebook in Google Colab. You can view them directly or make a copy to run and modify yourself.

---

### ▶️ `01_shor_n15`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1sBf3nyACTdpnbdwpm-SnqV1_KtBmWeVM?usp=sharing)

- Focused on factoring \( N = 15 \) with \( a = 7 \)
- Builds and compares quantum circuits using different methods
- Simulates both **ideal** and **noisy** hardware
- Demonstrates tradeoffs between circuit complexity and error rates

---

### ▶️ `02_shor_largerN`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/12GVp5INgQkzvtxAxOe5UirsOMJ2ukQuh?usp=sharing)

- Generalizes Shor’s algorithm to factor arbitrary values of \( N \)
- Explores challenges when scaling beyond \( N = 15 \)
- Demonstrates circuit depth and qubit limitations
- Runs only on ideal simulators due to size and noise sensitivity

---

### ▶️ `00_shor_n15_standard`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rkwGLca1b-1Yj_9WUru1sdVEtfXOW_07?usp=sharing)

- Classical exploration of the \( a^x \mod N \) function
- Demonstrates periodicity using Fourier transforms
- No quantum simulation; builds intuition for how the algorithm works

---

## ✅ How to Run

You don’t need to install anything. Just open a notebook in Google Colab, save a copy to your own Drive, and run the cells.

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code with attribution.
