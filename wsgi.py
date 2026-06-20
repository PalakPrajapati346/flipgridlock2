from src.app import app

# This is what Gunicorn looks for
if __name__ == "__main__":
    app.run()