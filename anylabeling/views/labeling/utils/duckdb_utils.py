import duckdb


class DuckDB:
    def __init__(self, database: str = ":memory:"):
        self.database = database
        self.connection = duckdb.connect(database)
        self.create_table()

    def create_table(self):
        self.connection.execute("CREATE TABLE images (filepath VARCHAR, timestamp TIMESTAMP, sample_id VARCHAR, "
                                "is_reviewed BOOLEAN, objective VARCHAR, illumination_type VARCHAR)")

    def insert_data(self, filepath: str, timestamp: str, sample_id: str, is_reviewed: bool, objective: str,
                    illumination_type: str):
        self.connection.execute("INSERT INTO images VALUES (?, ?, ?, ?, ?, ?)",
                                (filepath, timestamp, sample_id, is_reviewed, objective, illumination_type))

    def select_data(self, query):
        self.connection.execute(query)
        return self.connection.fetchall()

    def get_all_data(self):
        return self.select_data("SELECT * FROM images")

    def get_column_values(self, column_name):
        values = self.connection.execute(f"SELECT {column_name} FROM images GROUP BY {column_name}").fetchall()
        return [value[0] for value in values]

    def update_value_by_column(self, filepath_to_change, change_column, change_value):
        self.connection.execute(f"UPDATE images SET {change_column} = ? WHERE filepath = ?",
                                (change_value, filepath_to_change))

    def get_values_by_filter(self, filters: list):
        query = "SELECT filepath FROM images WHERE "
        query += " AND ".join([f"{f[0]} {f[1]} {f[2]}" for f in filters])
        result = self.connection.execute(query).fetchall()
        return [r[0] for r in result]

    def close_connection(self):
        self.connection.close()


def main():
    db = DuckDB()
    db.insert_data('/path/to/image.jpg', '2021-09-01 12:01:03', '1', True, 'x50', 'backlight')
    db.insert_data('/path/to/image.jpg', '2021-09-02 12:01:03', '2', False, 'x100', 'frontlight')
    print(db.select_data("SELECT * FROM images"))
    print(db.get_column_values('objective'))
    db.close_connection()


if __name__ == '__main__':
    main()
