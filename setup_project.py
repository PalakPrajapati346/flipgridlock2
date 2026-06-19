files = {
    'README.md': '# Gridlock 2.0: AI Traffic Command Center\n\n## Instructions to Run\n1. **Setup**: pip install -r requirements.txt\n2. **Run**: python app.py\n3. **Access**: http://127.0.0.1:5000\n\n## Technical Integrity\nThis project follows a strict **leakage-free** training pipeline, explicitly excluding duration_minutes and status to ensure pre-event honesty.',
    '.gitignore': 'venv/\n__pycache__/\n*.pyc\n.ipynb_checkpoints/\n*.joblib\n',
    'app.py': 'from flask import Flask\napp = Flask(__name__)\n\n@app.route("/")\ndef home():\n    return "Gridlock 2.0 Command Center Running"\n\nif __name__ == "__main__":\n    app.run(debug=True)',
    'requirements.txt': 'flask\nscikit-learn\nlightgbm\npandas\numpy\njoblib'
}
for f, c in files.items():
    with open(f, 'w', encoding='utf-8') as file:
        file.write(c)
