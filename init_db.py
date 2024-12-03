from app import app, db  # Replace 'app' with the name of your Flask app file without the .py extension

with app.app_context():
    db.create_all()
    print("Database and tables created successfully.")
