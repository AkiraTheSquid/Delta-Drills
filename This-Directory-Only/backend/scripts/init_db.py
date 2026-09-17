from sqlalchemy import text

from app.db import Base, engine
from app import models  # noqa: F401


def main() -> None:
    with engine.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS "pgcrypto";'))
        conn.commit()
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")


if __name__ == "__main__":
    main()
