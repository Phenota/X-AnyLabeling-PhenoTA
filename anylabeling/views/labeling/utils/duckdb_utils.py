import os
from datetime import datetime
from pathlib import Path
from time import mktime

import duckdb

from anylabeling.views.labeling.utils.img_metadata_utils import update_metadata_to_jpg_file


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


def generate_files_with_metadata(src_dir, dst_dir):
    """
    Code to duplicate src_dir with many cell images to 30 folders so we get 1000 images in 30 folders
    During duplicate we also set timestamp and metadata to the images
    """
    OBJECTIVES = ['x50', 'x100']
    ILLUMINATION_TYPES = ['backlight', 'frontlight']

    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    src_img_files = [f for f in src_path.glob('**/*.jpg') if f.is_file()]
    for i in range(30):
        sub_folder = dst_path / f"folder_{i+1:02}"
        sub_folder.mkdir(parents=True, exist_ok=True)
        for img_file in src_img_files:
            img_file_dst = sub_folder / img_file.name
            img_file_dst.write_bytes(img_file.read_bytes())
            metadata = dict(
                sample_id=f"SAMPLE_ID_{i+1:02}",
                objective=OBJECTIVES[i % len(OBJECTIVES)],
                illumination_type=ILLUMINATION_TYPES[i % len(ILLUMINATION_TYPES)]
            )
            update_metadata_to_jpg_file(str(img_file_dst), metadata)
            ts = datetime(2024, 8, i+1, 9, 9, 9)
            ts_utime = mktime(ts.timetuple())
            os.utime(str(img_file_dst), (ts_utime, ts_utime))



if __name__ == '__main__':
    main()
    # generate_files_with_metadata(r"c:\Users\yoram\Downloads\detector",
    #                              r"c:\Users\yoram\Downloads\hier_images")