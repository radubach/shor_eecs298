# shor_eecs298

# Shor's Algorithm – Group Project

This repository is for our group project on **Shor’s Algorithm** for our quantum computing class EECS298. We'll use this space to share code, collaborate, and track our progress.

---

## 🧠 Project Overview

Shor’s algorithm is a quantum algorithm that efficiently factors large integers—something classical algorithms struggle with. Our goals include:

- Understanding the math behind Shor's algorithm
- Implementing it using Python and Qiskit (or another framework)
- Demonstrating it on small numbers (e.g., factoring 15 into 3 × 5)
- Explaining its implications for cryptography

---

## Colab Notebooks

Notebook that runs qiskit code [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.
google.com/drive/1sBf3nyACTdpnbdwpm-SnqV1_KtBmWeVM#scrollTo=0mhjzedT1Q62)

Notebook that runs demo on standard computing [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rkwGLca1b-1Yj_9WUru1sdVEtfXOW_07#scrollTo=cf1xbBavpM6s)

## 🛠️ Getting Started

To use this repo:

1. Clone the repository:

   ```bash
   git clone https://github.com/radubach/shor_eecs298.git
   cd shor_eecs298
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

> If we don't have a `requirements.txt` yet, we can add it later as we decide what libraries we’re using.

---

## 👨‍💻 Recommended Git Workflow

Don't worry if you're new to Git — version control helps us experiment safely. If something goes wrong, we can always roll back to a previous version.

Here’s a typical workflow:

```bash
# Step 0: Make sure you're up to date with main
git checkout main
git pull

# Step 1: Create a branch for your work
git checkout -b your_branch_name

# Step 2: Write your code and save your files

# Step 3: Stage your changes
git add .

# Step 4: Commit your changes with a message
git commit -m "describe your changes"

# Step 5: (Optional) Make more changes and commit again

# Step 6: Push your branch to GitHub
git push origin your_branch_name
```

### Step 7: Open a Pull Request

Once your branch is pushed, go to the repo on GitHub website:

- Click **"Compare & pull request"**
- Write a short description of what you did
- (Optional) Ask a teammate to review before merging into `main`

---

## 💡 Tips

- **Feel free to experiment!** Version control means nothing is permanent — we can undo or fix anything.
- **If you get stuck**, ask in the group chat or leave a comment in the repo.
- We can use **Issues** to track bugs or questions and the **Projects** tab if we want a kanban-style board.

---

## 📂 Suggested Folder Structure

```
shors-algorithm-group-project/
│
├── README.md
├── requirements.txt
├── src/               # Python code goes here
│   └── shor.py        # Starting point for the algorithm
├── notebooks/         # Jupyter notebooks and exploration
└── docs/              # Notes, diagrams, and explanations
```

---

Let’s build something cool 🔬⚛️
